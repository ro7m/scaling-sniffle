class BoundingBox {
  final int? id;
  final double x;
  final double y;
  final double width;
  final double height;
  final List<List<double>>? coordinates;
  final Map<String, dynamic>? config;

  BoundingBox({
    this.id,
    required this.x,
    required this.y,
    required this.width,
    required this.height,
    this.coordinates,
    this.config,
  });