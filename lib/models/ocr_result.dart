class OCRResult {
  final String text;
  final BoundingBox boundingBox;

  OCRResult({
    required this.text,
    required this.boundingBox,
  });

  Map<String, dynamic> toJson() => {
    'text': text,
    'boundingBox': {
      'id': boundingBox.id,
      'coordinates': boundingBox.coordinates,
      'config': boundingBox.config,
      'x': boundingBox.x,
      'y': boundingBox.y,
      'width': boundingBox.width,
      'height': boundingBox.height,
    },
  };
}