import 'package:flutter/material.dart';
import 'package:scaling-sniffle/models/ocr_result.dart';

class BoundingBoxPainter extends CustomPainter {
  final List<OCRResult> results;

  BoundingBoxPainter(this.results);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.red
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    for (final result in results) {
      final rect = Rect.fromLTWH(
        result.boundingBox.x,
        result.boundingBox.y,
        result.boundingBox.width,
        result.boundingBox.height,
      );
      canvas.drawRect(rect, paint);
    }
  }

  @override
  bool shouldRepaint(BoundingBoxPainter oldDelegate) {
    return results != oldDelegate.results;
  }
}
