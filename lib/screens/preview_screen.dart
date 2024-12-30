import 'package:flutter/material.dart';
import 'dart:ui' as ui;
import 'package:camera/camera.dart';
import '../services/ocr_service.dart';
import '../models/bounding_box.dart';

class PreviewScreen extends StatefulWidget {
  final XFile image;

  PreviewScreen({required this.image});

  @override
  _PreviewScreenState createState() => _PreviewScreenState();
}

class _PreviewScreenState extends State<PreviewScreen> {
  final OCRService _ocrService = OCRService();
  List<BoundingBox> _boundingBoxes = [];
  String _extractedText = '';
  ui.Image? _decodedImage;
  Size? _imageSize;
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _ocrService.setDebugCallback = _addDebugMessage;
    _processImage();
  }

  void _addDebugMessage(String message) {
    print(message);
  }

  Future<void> _processImage() async {
    try {
      await _ocrService.loadModels(debugCallback: _addDebugMessage);
      final imageBytes = await widget.image.readAsBytes();
      final ui.Codec codec = await ui.instantiateImageCodec(imageBytes);
      final ui.FrameInfo frameInfo = await codec.getNextFrame();
      _decodedImage = frameInfo.image;
      
      // Store original image dimensions
      _imageSize = Size(_decodedImage!.width.toDouble(), _decodedImage!.height.toDouble());
      
      final results = await _ocrService.processImage(_decodedImage!, debugCallback: _addDebugMessage);
      
      if (mounted) {
        setState(() {
          _boundingBoxes = results.map((r) => r.boundingBox).toList();
          _extractedText = results.map((r) => r.text).join('\n');
          _isLoading = false;
        });
      }
    } catch (e) {
      print('Error processing image: $e');
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('OCR Preview'),
      ),
      body: _isLoading 
          ? Center(child: CircularProgressIndicator())
          : _buildPreviewContent(),
    );
  }

  Widget _buildPreviewContent() {
    if (_decodedImage == null || _imageSize == null) {
      return Center(child: Text('Failed to load image'));
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        // Calculate scaling to fit the screen while maintaining aspect ratio
        final double screenAspectRatio = constraints.maxWidth / constraints.maxHeight;
        final double imageAspectRatio = _imageSize!.width / _imageSize!.height;
        
        late double scaledWidth;
        late double scaledHeight;
        
        if (screenAspectRatio > imageAspectRatio) {
          // Screen is wider than image
          scaledHeight = constraints.maxHeight;
          scaledWidth = scaledHeight * imageAspectRatio;
        } else {
          // Screen is taller than image
          scaledWidth = constraints.maxWidth;
          scaledHeight = scaledWidth / imageAspectRatio;
        }

        return Center(
          child: SizedBox(
            width: scaledWidth,
            height: scaledHeight,
            child: Stack(
              children: [
                // Original image
                Image.file(
                  File(widget.image.path),
                  width: scaledWidth,
                  height: scaledHeight,
                  fit: BoxFit.fill,
                ),
                // Bounding boxes overlay
                CustomPaint(
                  size: Size(scaledWidth, scaledHeight),
                  painter: BoundingBoxPainter(
                    boundingBoxes: _boundingBoxes,
                    scale: Size(
                      scaledWidth / _imageSize!.width,
                      scaledHeight / _imageSize!.height,
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

class BoundingBoxPainter extends CustomPainter {
  final List<BoundingBox> boundingBoxes;
  final Size scale;

  BoundingBoxPainter({
    required this.boundingBoxes,
    required this.scale,
  });

  @override
  void paint(Canvas canvas, Size size) {
    for (var box in boundingBoxes) {
      final paint = Paint()
        ..color = _parseColor(box.config['stroke'] ?? '#FF0000')
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0;

      // Convert normalized coordinates to actual pixels
      final points = box.coordinates.map((coord) => Offset(
        coord[0] * size.width,
        coord[1] * size.height,
      )).toList();

      // Draw the box
      final path = Path()
        ..moveTo(points[0].dx, points[0].dy)
        ..lineTo(points[1].dx, points[1].dy)
        ..lineTo(points[2].dx, points[2].dy)
        ..lineTo(points[3].dx, points[3].dy)
        ..close();

      canvas.drawPath(path, paint);
    }
  }

  Color _parseColor(String hexColor) {
    try {
      hexColor = hexColor.replaceAll('#', '');
      if (hexColor.length == 6) {
        return Color(int.parse('FF$hexColor', radix: 16));
      }
      return Colors.red;
    } catch (e) {
      return Colors.red;
    }
  }

  @override
  bool shouldRepaint(BoundingBoxPainter oldDelegate) =>
      boundingBoxes != oldDelegate.boundingBoxes || scale != oldDelegate.scale;
}
