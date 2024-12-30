class BoundingBoxPainter extends CustomPainter {
  final ui.Image image;
  final List<BoundingBox> boundingBoxes;
  final Color Function(String) parseColor; // Add color parser function

  BoundingBoxPainter({
    required this.image,
    required this.boundingBoxes,
    required this.parseColor, // Required in constructor
  });

  @override
  void paint(Canvas canvas, Size size) {
    final imageSize = Size(image.width.toDouble(), image.height.toDouble());
    final fitSize = _calculateFitSize(imageSize, size);
    final rect = _centerRect(fitSize, size);
    canvas.drawImage(image, rect.topLeft, Paint());

    for (var box in boundingBoxes) {
      final paint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0
        ..color = parseColor(box.config?['stroke'] ?? '#FF0000');

      if (box.coordinates != null) {
        final points = box.coordinates!.map((coord) => Offset(
              coord[0] * rect.width + rect.left,
              coord[1] * rect.height + rect.top,
            )).toList();

        final path = Path();
        path.moveTo(points[0].dx, points[0].dy);
        for (int i = 1; i < points.length; i++) {
          path.lineTo(points[i].dx, points[i].dy);
        }
        path.close();
        canvas.drawPath(path, paint);
      }
    }
  }

  Rect _centerRect(Size fitSize, Size size) {
    final left = (size.width - fitSize.width) / 2;
    final top = (size.height - fitSize.height) / 2;
    return Rect.fromLTWH(left, top, fitSize.width, fitSize.height);
  }

  Size _calculateFitSize(Size imageSize, Size boxSize) {
    final imageAspectRatio = imageSize.width / imageSize.height;
    final boxAspectRatio = boxSize.width / boxSize.height;

    late double fitWidth;
    late double fitHeight;

    if (imageAspectRatio > boxAspectRatio) {
      fitWidth = boxSize.width;
      fitHeight = fitWidth / imageAspectRatio;
    } else {
      fitHeight = boxSize.height;
      fitWidth = fitHeight * imageAspectRatio;
    }

    return Size(fitWidth, fitHeight);
  }

  @override
  bool shouldRepaint(BoundingBoxPainter oldDelegate) {
    return oldDelegate.image != image || 
           oldDelegate.boundingBoxes != boundingBoxes;
  }
}