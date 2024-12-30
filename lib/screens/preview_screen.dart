import 'package:flutter/material.dart';
import 'dart:ui' as ui;
import 'dart:io';
import 'package:image_picker/image_picker.dart';
import '../services/ocr_service.dart';
import '../models/bounding_box.dart';
import '../services/bounding_box_painter.dart';

class PreviewScreen extends StatefulWidget {
  final XFile image;

  const PreviewScreen({Key? key, required this.image}) : super(key: key);

  @override
  _PreviewScreenState createState() => _PreviewScreenState();
}

class _PreviewScreenState extends State<PreviewScreen> {
  List<BoundingBox> _boundingBoxes = [];
  String _extractedText = '';
  ui.Image? _decodedImage;
  bool _isProcessing = true;
  String _debugText = '';

  @override
  void initState() {
    super.initState();
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

      // Load and decode the image
      final imageBytes = await widget.image.readAsBytes();
      final ui.Codec codec = await ui.instantiateImageCodec(imageBytes);
      final ui.FrameInfo frameInfo = await codec.getNextFrame();
      
      setState(() {
        _decodedImage = frameInfo.image;
      });

      // Process OCR here and update _boundingBoxes and _extractedText
      // Add your OCR processing logic here
      
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
    final size = MediaQuery.of(context).size;
    
    return Scaffold(
      appBar: AppBar(
        title: const Text('OCR Preview'),
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          if (_decodedImage != null)
            CustomPaint(
              painter: BoundingBoxPainter(
                image: _decodedImage!,
                boundingBoxes: _boundingBoxes,
                screenSize: size,
              ),
              size: size,
            )
          else if (!_isProcessing)
            Image.file(
              File(widget.image.path),
              fit: BoxFit.contain,
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