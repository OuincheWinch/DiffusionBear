// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "DiffusionBearMingHelper",
    platforms: [.macOS(.v26)],
    dependencies: [
        // Parity-gated MIT Ming-Image port. Pin the exact reviewed revision; the
        // helper only uses the core MingImage product, not the MLXEngine wrapper.
        .package(url: "https://github.com/xocialize/ming-image-swift.git", revision: "daf3b5bb222dc07f7f3adc596cb4567cf480541a"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.32.3"),
    ],
    targets: [
        .executableTarget(
            name: "MingHelper",
            dependencies: [
                .product(name: "MingImage", package: "ming-image-swift"),
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/MingHelper"
        )
    ]
)
