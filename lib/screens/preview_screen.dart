import 'package:flutter/material.dart';
import 'dart:ui' as ui; // Import dart:ui with an alias
import 'package:camera/camera.dart'; // Import Camera library
import '../services/ocr_service.dart';
import '../models/bounding_box.dart';
import '../services/bounding_box_painter.dart';

class PreviewScreen extends StatefulWidget {
  final XFile image; // Use XFile to match the captured image type

  PreviewScreen({required this.image});

  @override
  _PreviewScreenState createState() => _PreviewScreenState();
}

class _PreviewScreenState extends State<PreviewScreen> {
  final OCRService _ocrService = OCRService();
  List<BoundingBox> _boundingBoxes = [];
  String _extractedText = '';
  ui.Image? _decodedImage;

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
    final imageBytes = await widget.image.readAsBytes();
    final ui.Codec codec = await ui.instantiateImageCodec(imageBytes);
    final ui.FrameInfo frameInfo = await codec.getNextFrame();
    _decodedImage = frameInfo.image; // Decode the image
    final results = await _ocrService.processImage(_decodedImage!, debugCallback: _addDebugMessage);
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
        child: _boundingBoxes.isEmpty || _decodedImage == null
            ? CircularProgressIndicator()
            : Column(
                children: [
                  Expanded(
                    child: CustomPaint(
                      painter: BoundingBoxPainter(
                        image: _decodedImage!, // Pass the decoded image
                        boundingBoxes: _boundingBoxes,
                      ),
                      child: Container(),
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: Text(
                      _extractedText,
                      style: TextStyle(fontSize: 16),
                    ),
                  ),
                ],
              ),
      ),
    );
  }
}