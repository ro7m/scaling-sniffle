// lib/models/bounding_box.dart
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

  // Add a copyWith method for transformations
  BoundingBox copyWith({
    int? id,
    double? x,
    double? y,
    double? width,
    double? height,
    List<List<double>>? coordinates,
    Map<String, dynamic>? config,
  }) {
    return BoundingBox(
      id: id ?? this.id,
      x: x ?? this.x,
      y: y ?? this.y,
      width: width ?? this.width,
      height: height ?? this.height,
      coordinates: coordinates ?? this.coordinates,
      config: config ?? this.config,
    );
  }
}