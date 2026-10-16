import Foundation
var failures = 0
func check(_ label: String, _ ok: Bool, _ detail: String = "") {
    print("  \(ok ? "PASS" : "FAIL")  \(label)\(detail.isEmpty ? "" : "  -> \(detail)")")
    if !ok { failures += 1 }
}
// 8001 is held by a fake answering 404; 9025 is the REAL DiffusionBear backend.
check("our backend on 9025 IS identified as ours", PortProbe.answersAsOurs(9025))
check("the fake on 8001 is NOT identified as ours", !PortProbe.answersAsOurs(8001),
      "answers 404")
let chosen = PortProbe.choosePort()
check("choosePort REUSES 9025 rather than stepping past it", chosen == 9025,
      "got \(chosen.map(String.init) ?? "nil") -- must not start a second engine")
print(failures == 0 ? "ALL PASS" : "\(failures) FAILED")
exit(failures == 0 ? 0 : 1)
