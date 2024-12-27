import 'dart:math' as math;

class BoundingBox {
  final double x;
  final double y;
  final double width;
  final double height;

  BoundingBox({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
  });

double clamp(double number, double size) {
  return math.max(0, math.min(number, size));
}

  BoundingBox transformBoundingBox(BoundingBox contour, int id, List<int> size) {
  double offset = (contour.width * contour.height * 1.8) / (2 * (contour.width + contour.height));
  double p1 = clamp(contour.x - offset, size[1].toDouble()) - 1;
  double p2 = clamp(p1 + contour.width + 2 * offset, size[1].toDouble()) - 1;
  double p3 = clamp(contour.y - offset, size[0].toDouble()) - 1;
  double p4 = clamp(p3 + contour.height + 2 * offset, size[0].toDouble()) - 1;

  return BoundingBox(
    x: p1 / size[1],
    y: p3 / size[0],
    width: (p2 - p1) / size[1],
    height: (p4 - p3) / size[0],
  );
}
}