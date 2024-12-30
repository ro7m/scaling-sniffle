import 'package:flutter/material.dart';
import 'dart:ui' as ui;
import 'dart:io';
import 'package:camera/camera';
import '../services/ocr_service.dart';
import '../models/bounding_box.dart';
import '../services/bounding_box_painter.dart';

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
  bool _isProcessing = true;
  String _debugText = '';

  @override
  void initState() {
    super.initState();
    _ocrService.setDebugCallback = _addDebugMessage;
    _processImage();
  }

  void _addDebugMessage(String message) {
    setState(() {
      _debugText += '$message\n';
    });
    print(message);
  }

  Future<void> _processImage() async {
    try {
      setState(() {
        _isProcessing = true;
      });

      await _ocrService.loadModels(debugCallback: _addDebugMessage);
      
      // Load and decode the image
      final imageBytes = await widget.image.readAsBytes();
      final ui.Codec codec = await ui.instantiateImageCodec(imageBytes);
      final ui.FrameInfo frameInfo = await codec.getNextFrame();
      setState(() {
        _decodedImage = frameInfo.image;
      });

      if (_decodedImage != null) {
        final results = await _ocrService.processImage(_decodedImage!, debugCallback: _addDebugMessage);
        setState(() {
          _boundingBoxes = results.map((r) => r.boundingBox).toList();
          _extractedText = results.map((r) => r.text).join('\n');
        });
      }
    } catch (e, stackTrace) {
      _addDebugMessage('Error: $e\n$stackTrace');
    } finally {
      setState(() {
        _isProcessing = false;
      });
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
      return Colors.red; // Default color
    } catch (e) {
      return Colors.red; // Default color on error
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('OCR Preview'),
      ),
      body: Stack(
        children: [
          if (_decodedImage != null)
            Center(
              child: CustomPaint(
                painter: BoundingBoxPainter(
                  image: _decodedImage!,
                  boundingBoxes: _boundingBoxes,
                ),
                size: Size(
                  MediaQuery.of(context).size.width,
                  MediaQuery.of(context).size.height,
                ),
              ),
            )
          else
            Center(
              child: Image.file(
                File(widget.image.path), // Now File is properly imported
                fit: BoxFit.contain,
              ),
            ),
          if (_isProcessing)
            const Center(
              child: CircularProgressIndicator(),
            ),
          // Display extracted text
          if (_extractedText.isNotEmpty)
            Positioned(
              bottom: 20,
              left: 20,
              right: 20,
              child: Container(
                padding: const EdgeInsets.all(8),
                color: Colors.black.withOpacity(0.7),
                child: Text(
                  _extractedText,
                  style: const TextStyle(color: Colors.white),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class BoundingBoxPainter extends CustomPainter {
  final ui.Image image;
  final List<BoundingBox> boundingBoxes;

  BoundingBoxPainter({
    required this.image,
    required this.boundingBoxes,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Draw the image
    final imageSize = Size(image.width.toDouble(), image.height.toDouble());
    final fitSize = _calculateFitSize(imageSize, size);
    final rect = _centerRect(fitSize, size);
    canvas.drawImage(image, rect.topLeft, Paint());

    // Draw bounding boxes
    for (var box in boundingBoxes) {
      final paint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0
        ..color = _parseColor(box.config?['stroke'] ?? '#FF0000');

      if (box.coordinates != null) {
        final points = box.coordinates!.map((coord) => Offset(
              coord[0] * rect.width + rect.left,
              coord[1] * rect.height + rect.top,
            )).toList();

        // Draw the box
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
    return oldDelegate.image != image || oldDelegate.boundingBoxes != boundingBoxes;
  }
}