import 'package:flutter/material.dart';
import 'dart:ui' as ui; // Import dart:ui with an alias
import '../models/bounding_box.dart';

class BoundingBoxPainter extends CustomPainter {
  final ui.Image image;
  final List<BoundingBox> boundingBoxes;

  BoundingBoxPainter({required this.image, required this.boundingBoxes});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.red
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;

    canvas.drawImage(image, Offset.zero, Paint());

    for (var box in boundingBoxes) {
      final rect = Rect.fromLTWH(
        box.x * size.width,
        box.y * size.height,
        box.width * size.width,
        box.height * size.height,
      );
      canvas.drawRect(rect, paint);
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) {
    return false;
  }
}