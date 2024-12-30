import 'package:flutter/material.dart';
import 'dart:ui' as ui; // Import dart:ui with an alias
import '../models/bounding_box.dart';

class BoundingBoxPainter extends CustomPainter {
  final ui.Image image;
  final List<BoundingBox> boundingBoxes;
  final Size screenSize;

  BoundingBoxPainter({
    required this.image,
    required this.boundingBoxes,
    required this.screenSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Calculate scale factors to fit image to screen while maintaining aspect ratio
    double scaleX = screenSize.width / image.width;
    double scaleY = screenSize.height / image.height;
    double scale = scaleX < scaleY ? scaleX : scaleY;

    // Calculate centered position
    double left = (screenSize.width - (image.width * scale)) / 2;
    double top = (screenSize.height - (image.height * scale)) / 2;

    // Draw the image
    final rect = Rect.fromLTWH(left, top, image.width * scale, image.height * scale);
    canvas.drawImage(image, rect.topLeft, Paint());

    // Draw bounding boxes
    for (var box in boundingBoxes) {
      if (box.coordinates != null) {
        final paint = Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2.0
          ..color = _parseColor(box.config?['stroke'] ?? '#FF0000');

        final path = Path();
        bool isFirst = true;
        
        for (var coord in box.coordinates!) {
          final double x = (coord[0] * image.width * scale) + left;
          final double y = (coord[1] * image.height * scale) + top;
          
          if (isFirst) {
            path.moveTo(x, y);
            isFirst = false;
          } else {
            path.lineTo(x, y);
          }
        }
        
        path.close();
        canvas.drawPath(path, paint);
      }
    }
  }

  Color _parseColor(String colorStr) {
    try {
      if (colorStr.startsWith('#')) {
        colorStr = colorStr.substring(1);
      }
      if (colorStr.length == 6) {
        return Color(int.parse('FF$colorStr', radix: 16));
      }
      return Colors.red;
    } catch (e) {
      return Colors.red;
    }
  }

  @override
  bool shouldRepaint(BoundingBoxPainter oldDelegate) {
    return oldDelegate.image != image || 
           oldDelegate.boundingBoxes != boundingBoxes ||
           oldDelegate.screenSize != screenSize;
  }
}