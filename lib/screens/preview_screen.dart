import 'package:flutter/material.dart';
import 'dart:ui' as ui;
import 'dart:io';
import 'package:image_picker/image_picker.dart';
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
                  parseColor: _parseColor, // Pass the color parser function
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
                File(widget.image.path),
                fit: BoxFit.contain,
              ),
            ),
          if (_isProcessing)
            const Center(
              child: CircularProgressIndicator(),
            ),
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