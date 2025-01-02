import 'package:flutter/material.dart';
import 'dart:ui' as ui;
import 'package:camera/camera.dart';
import '../services/ocr_service.dart';
import '../models/bounding_box.dart';
import '../services/bounding_box_painter.dart';
import '../models/ocr_result.dart';

class PreviewScreen extends StatefulWidget {
  final XFile image;

  const PreviewScreen({Key? key, required this.image}) : super(key: key);

  @override
  _PreviewScreenState createState() => _PreviewScreenState();
}

class _PreviewScreenState extends State<PreviewScreen> {
  final OCRService _ocrService = OCRService();
  List<OCRResult> _results = [];
  bool _isProcessing = true;
  String _errorMessage = '';

  @override
  void initState() {
    super.initState();
    _processImage();
  }

  Future<void> _processImage() async {
    try {
      await _ocrService.loadModels();
      final results = await _ocrService.processImage(widget.image);
      setState(() {
        _results = results;
        _isProcessing = false;
      });
    } catch (e) {
      setState(() {
        _errorMessage = 'Error processing image: ${e.toString()}';
        _isProcessing = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Extracted Text'),
      ),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (_isProcessing) {
      return const Center(
        child: CircularProgressIndicator(),
      );
    }

    if (_errorMessage.isNotEmpty) {
      return Center(
        child: Text(_errorMessage, style: const TextStyle(color: Colors.red)),
      );
    }

    if (_results.isEmpty) {
      return const Center(
        child: Text('No text detected'),
      );
    }

    return Stack(
      children: [
        // Image
        Image.file(File(widget.image.path)),
        
        // Overlay texts at their positions
        ..._buildOverlayTexts(),
      ],
    );
  }

  List<Widget> _buildOverlayTexts() {
    // Group texts by their vertical position (y-coordinate)
    final Map<int, List<OCRResult>> lineGroups = {};
    const int lineThreshold = 10; // Pixels threshold to consider texts on same line

    for (var result in _results) {
      bool addedToExistingLine = false;
      for (var y in lineGroups.keys) {
        if ((result.boundingBox.y - y).abs() <= lineThreshold) {
          lineGroups[y]!.add(result);
          addedToExistingLine = true;
          break;
        }
      }
      if (!addedToExistingLine) {
        lineGroups[result.boundingBox.y] = [result];
      }
    }

    // Sort texts within each line by x-coordinate
    for (var line in lineGroups.values) {
      line.sort((a, b) => a.boundingBox.x.compareTo(b.boundingBox.x));
    }

    // Create positioned texts
    return lineGroups.entries.map((entry) {
      final line = entry.value;
      return Positioned(
        left: line.first.boundingBox.x,
        top: line.first.boundingBox.y,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
          color: Colors.black.withOpacity(0.5),
          child: Text(
            line.map((result) => result.text).join(' '),
            style: const TextStyle(
              color: Colors.white,
              fontSize: 16,
            ),
          ),
        ),
      );
    }).toList();
  }
}