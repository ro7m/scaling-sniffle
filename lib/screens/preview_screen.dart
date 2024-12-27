import 'package:flutter/material.dart';
import '../services/ocr_service.dart';
import '../services/bounding_box_painter.dart';

class PreviewScreen extends StatefulWidget {
  final ui.Image image;

  PreviewScreen({required this.image});

  @override
  _PreviewScreenState createState() => _PreviewScreenState();
}

class _PreviewScreenState extends State<PreviewScreen> {
  final OCRService _ocrService = OCRService();
  List<BoundingBox> _boundingBoxes = [];
  String _extractedText = '';

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
    await _ocrService.loadModels(debugCallback: _addDebugMessage);
    final results = await _ocrService.processImage(widget.image, debugCallback: _addDebugMessage);
    setState(() {
      _boundingBoxes = results.map((r) => r.boundingBox).toList();
      _extractedText = results.map((r) => r.text).join('\n');
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('OCR Preview'),
      ),
      body: Center(
        child: CustomPaint(
          painter: BoundingBoxPainter(
            image: widget.image,
            boundingBoxes: _boundingBoxes,
          ),
          child: Container(),
        ),
      ),
    );
  }
}