import CoreGraphics
import Foundation
import ImageIO
import MLX
import MingImage
import UniformTypeIdentifiers

struct Options {
    var root: URL
    var prompt: String
    var output: URL
    var width = 1024
    var height = 1024
    var steps = 12
    var guidance: Float = 1.0
    var seed: UInt64 = 42
    var resolution = 1024
    var transparent = false
    var cacheGB = 2
    var dtype = "bf16"
}

enum HelperError: Error, CustomStringConvertible {
    case usage(String)
    case pngEncode
    var description: String {
        switch self {
        case .usage(let message): return message
        case .pngEncode: return "PNG encoding failed"
        }
    }
}

func jsonEscape(_ value: String) -> String {
    var out = ""
    for scalar in value.unicodeScalars {
        switch scalar {
        case "\\": out += "\\\\"
        case "\"": out += "\\\""
        case "\n": out += "\\n"
        case "\r": out += "\\r"
        case "\t": out += "\\t"
        default: out.unicodeScalars.append(scalar)
        }
    }
    return out
}

func emit(_ fields: [(String, String)]) {
    let body = fields.map { "\"\($0.0)\":\($0.1)" }.joined(separator: ",")
    print("{\(body)}")
    fflush(stdout)
}

func emitPhase(_ phase: String, _ detail: String = "") {
    emit([("phase", "\"\(jsonEscape(phase))\""), ("detail", "\"\(jsonEscape(detail))\"")])
}

func emitProgress(step: Int, total: Int, phase: String = "denoise") {
    emit([
        ("phase", "\"\(phase)\""),
        ("progress", "{\"step\":\(step),\"total\":\(total)}"),
    ])
}

func parseOptions() throws -> Options {
    var args = Array(CommandLine.arguments.dropFirst())
    func value(_ name: String) -> String? {
        guard let index = args.firstIndex(of: name), index + 1 < args.count else { return nil }
        let result = args[index + 1]
        args.removeSubrange(index...(index + 1))
        return result
    }
    func flag(_ name: String) -> Bool {
        guard let index = args.firstIndex(of: name) else { return false }
        args.remove(at: index)
        return true
    }

    guard let root = value("--root"), let prompt = value("--prompt"), let output = value("--output") else {
        throw HelperError.usage("required: --root <snapshot> --prompt <text> --output <png> [--width 1024] [--height 1024] [--steps 12] [--guidance 1.0] [--seed 42] [--resolution 1024|2048] [--transparent]")
    }
    var options = Options(root: URL(fileURLWithPath: root), prompt: prompt, output: URL(fileURLWithPath: output))
    if let v = value("--width"), let parsed = Int(v) { options.width = parsed }
    if let v = value("--height"), let parsed = Int(v) { options.height = parsed }
    if let v = value("--steps"), let parsed = Int(v) { options.steps = parsed }
    if let v = value("--guidance"), let parsed = Float(v) { options.guidance = parsed }
    if let v = value("--seed"), let parsed = UInt64(v) { options.seed = parsed }
    if let v = value("--resolution"), let parsed = Int(v) { options.resolution = parsed }
    if let v = value("--cache-gb"), let parsed = Int(v) { options.cacheGB = parsed }
    if let v = value("--dtype") {
        guard v == "bf16" || v == "fp16" else { throw HelperError.usage("--dtype must be bf16 or fp16") }
        options.dtype = v
    }
    options.transparent = flag("--transparent")
    if !args.isEmpty { throw HelperError.usage("unknown arguments: \(args.joined(separator: " "))") }
    if options.width < 256 || options.height < 256 || options.steps < 1 {
        throw HelperError.usage("invalid dimensions or step count")
    }
    return options
}

func encodePNG(rgba: [UInt8], width: Int, height: Int) throws -> Data {
    guard let provider = CGDataProvider(data: Data(rgba) as CFData),
          let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                              bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                              provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    else { throw HelperError.pngEncode }
    let out = NSMutableData()
    guard let destination = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil)
    else { throw HelperError.pngEncode }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else { throw HelperError.pngEncode }
    return out as Data
}

func alphaIsTransparent(_ rgba: [UInt8]) -> Bool {
    let count = rgba.count / 4
    guard count > 0 else { return false }
    var clear = 0
    for index in stride(from: 3, to: count * 4, by: 4) where rgba[index] <= 12 { clear += 1 }
    return Double(clear) / Double(count) > 0.10
}

func floorAlpha(_ rgba: inout [UInt8]) {
    for index in stride(from: 3, to: rgba.count, by: 4) where rgba[index] <= 25 { rgba[index] = 0 }
}

func transparentPrompts(_ prompt: String) -> [String] {
    [
        "transparent canvas, not white, not checkerboard, \(prompt)",
        "不要白底，不要棋盘格，只要透明通道, \(prompt)",
    ]
}

func run() async throws {
    let options = try parseOptions()
    MLX.Memory.cacheLimit = options.cacheGB << 30

    let resolution = MingTask.textToImage.resolve(options.resolution)
    let sizes = MingSizes.processRatio(oriH: options.height, oriW: options.width, highres: resolution)
    let diffusion = sizes.closest
    // DiffusionBear dimensions are exact output dimensions; `sizes.resize` is only
    // the bucket-preserving intermediate aspect and can exceed the requested preset.
    let output = MingSize(height: options.height, width: options.width)

    emitPhase("encoding_prompt", "Stage 1/3: loading Ming conditioner and encoding prompt...")
    let prompts = options.transparent ? transparentPrompts(options.prompt) : [options.prompt]
    let ids = try await MingT2IPipeline.promptIds(prompts, root: options.root)
    // Pre-quantized snapshots are detected from the sibling .scales/.biases keys;
    // the conditioner is scoped inside encodeAll and released before the DiT loads.
    let computeDtype: DType = options.dtype == "fp16" ? .float16 : .bfloat16
    let conditions = try MingT2IPipeline.encodeAll(promptIds: ids, root: options.root, dtype: computeDtype)
    emitPhase("loading_transformer", "Stage 2/3: loading quantized Ming DiT and RGBA VAE...")

    let pipeline = try MingT2IPipeline(root: options.root, ditDtype: computeDtype, vaeDtype: .float32)
    let noise = MingT2IPipeline.noise(width: diffusion.width, height: diffusion.height, seed: options.seed)
    var pixels: [UInt8] = []

    for (attempt, condition) in conditions.enumerated() {
        emitPhase("denoise", "Stage 2/3: denoising \(diffusion.width)×\(diffusion.height) latents...")
        let latents = try pipeline.denoise(condition, noise: noise, steps: options.steps, guidance: options.guidance) { step, total in
            emitProgress(step: step, total: total)
        }
        if any(isNaN(latents)).item(Bool.self) || any(isInf(latents)).item(Bool.self) {
            throw HelperError.usage("non-finite latents with \(options.dtype) DiT compute; use bf16")
        }
        emitPhase("decode", "Stage 3/3: decoding RGBA image...")
        var rendered = pipeline.decode(latents).rgba.asArray(UInt8.self)
        if output != diffusion {
            rendered = MingPIL.resizeRGBA(rendered, width: diffusion.width, height: diffusion.height,
                                          outWidth: output.width, outHeight: output.height, filter: .lanczos)
        }
        MLX.Memory.clearCache()
        if attempt == 0 { pixels = rendered }
        if !options.transparent || alphaIsTransparent(rendered) {
            pixels = rendered
            break
        }
    }
    if options.transparent { floorAlpha(&pixels) }

    let png = try encodePNG(rgba: pixels, width: output.width, height: output.height)
    try png.write(to: options.output, options: .atomic)
    emit([
        ("status", "\"success\""),
        ("path", "\"\(jsonEscape(options.output.path))\""),
        ("width", "\(output.width)"),
        ("height", "\(output.height)"),
        ("dtype", "\"\(options.dtype)\""),
        ("steps", "\(options.steps)"),
        ("guidance", "\(options.guidance)"),
    ])
}

@main
struct MingHelperMain {
    static func main() async {
        do {
            try await run()
        } catch {
            emit([("status", "\"error\""), ("error", "\"\(jsonEscape(String(describing: error)))\"")])
            exit(1)
        }
    }
}
