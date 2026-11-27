#!/bin/bash
# Build mlx.metallib for a SwiftPM-built MingHelper. SwiftPM can compile the C++
# runtime but cannot compile MLX's Metal shaders; without this colocated library
# the helper starts and immediately dies with "Failed to load the default metallib".
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
MLX_ROOT="$PACKAGE_DIR/.build/checkouts/mlx-swift/Source/Cmlx/mlx"
KERNELS="$MLX_ROOT/mlx/backend/metal/kernels"
if [ ! -d "$KERNELS" ]; then
  echo "mlx-swift Metal kernels not found at $KERNELS; run swift build first" >&2
  exit 1
fi

RELEASE_DIR="$PACKAGE_DIR/.build/arm64-apple-macosx/release"
[ -d "$RELEASE_DIR" ] || RELEASE_DIR="$PACKAGE_DIR/.build/release"
mkdir -p "$RELEASE_DIR"
OUT="${1:-$RELEASE_DIR/mlx.metallib}"
WORK="$PACKAGE_DIR/.build/ming-metallib"
rm -rf "$WORK"
mkdir -p "$WORK/air"

cat > "$WORK/compile_one.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
MLX_ROOT="$1"
FILE="$2"
WORK="$3"
cd "$MLX_ROOT/mlx/backend/metal/kernels"
NAME="$(echo "$FILE" | tr / _)"
NAME="${NAME%.metal}"
xcrun -sdk macosx metal -x metal -Wall -fno-fast-math \
  -Wno-c++17-extensions -Wno-c++20-extensions -mmacosx-version-min=26.0 \
  -c "$FILE" -I"$MLX_ROOT" -o "$WORK/air/$NAME.air"
EOF

cd "$KERNELS"
find . -name '*.metal' | sed 's|^\./||' | sort > "$WORK/list.txt"
if [ ! -s "$WORK/list.txt" ]; then
  echo "no MLX Metal shaders found in $KERNELS" >&2
  exit 1
fi
JOBS="${MING_METAL_JOBS:-6}"
while IFS= read -r shader; do
  printf '%s\0' "$shader"
done < "$WORK/list.txt" | xargs -0 -P "$JOBS" -n 1 -I{} bash "$WORK/compile_one.sh" "$MLX_ROOT" {} "$WORK"

xcrun -sdk macosx metal -mmacosx-version-min=26.0 "$WORK"/air/*.air -o "$OUT"
echo "built $OUT ($(wc -l < "$WORK/list.txt" | tr -d ' ') shader modules)"
