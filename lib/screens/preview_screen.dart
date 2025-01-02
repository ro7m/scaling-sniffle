import 'package:flutter/material.dart';
import 'dart:ui' as ui;
import 'dart:io'; 
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
      return const Center(child: CircularProgressIndicator());
    }

    if (_errorMessage.isNotEmpty) {
      return Center(
        child: Text(_errorMessage, style: const TextStyle(color: Colors.red)),
      );
    }

    if (_results.isEmpty) {
      return const Center(child: Text('No text detected'));
    }

    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: _buildStructuredText(),
    );
  }

  Widget _buildStructuredText() {
    // Sort results by vertical position and group into lines
    final Map<int, List<OCRResult>> lineGroups = _groupIntoLines();
    
    // Convert to sorted list of lines
    final sortedLines = lineGroups.entries.toList()
      ..sort((a, b) => a.key.compareTo(b.key));

    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: sortedLines.map((line) {
          return _buildTextLine(line.value);
        }).toList(),
      ),
    );
  }

  Map<int, List<OCRResult>> _groupIntoLines() {
    final Map<int, List<OCRResult>> lineGroups = {};
    const int lineThreshold = 10;

    for (var result in _results) {
      final int yPos = result.boundingBox.y.round();
      bool addedToExistingLine = false;

      for (var y in lineGroups.keys) {
        if ((yPos - y).abs() <= lineThreshold) {
          lineGroups[y]!.add(result);
          addedToExistingLine = true;
          break;
        }
      }

      if (!addedToExistingLine) {
        lineGroups[yPos] = [result];
      }
    }

    // Sort each line by x-coordinate
    for (var line in lineGroups.values) {
      line.sort((a, b) => a.boundingBox.x.compareTo(b.boundingBox.x));
    }

    return lineGroups;
  }

  Widget _buildTextLine(List<OCRResult> lineResults) {
    // Calculate relative spacing between words
    final List<Widget> lineWidgets = [];
    
    for (int i = 0; i < lineResults.length; i++) {
      final result = lineResults[i];
      
      // Add spacing based on x-coordinates if not first element
      if (i > 0) {
        final previousX = lineResults[i - 1].boundingBox.x + lineResults[i - 1].boundingBox.width;
        final currentX = result.boundingBox.x;
        final gap = currentX - previousX;
        
        // Add spacing if gap is significant
        if (gap > 20) {  // Threshold for significant gap
          lineWidgets.add(SizedBox(width: 24));  // Standard spacing for gaps
        } else {
          lineWidgets.add(const SizedBox(width: 8));  // Normal word spacing
        }
      }

      // Add the text with styling based on its properties
      lineWidgets.add(
        Container(
          decoration: BoxDecoration(
            border: Border.all(color: Colors.grey.withOpacity(0.3)),
            borderRadius: BorderRadius.circular(4),
            color: Colors.grey.withOpacity(0.1),
          ),
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          child: Text(
            result.text,
            style: TextStyle(
              fontSize: 16,
              fontWeight: _determineFontWeight(result.boundingBox.height),
            ),
          ),
        ),
      );
    }

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Wrap(
        crossAxisAlignment: WrapCrossAlignment.center,
        children: lineWidgets,
      ),
    );
  }

  FontWeight _determineFontWeight(double height) {
    // Determine font weight based on text height (assuming larger text might be headers)
    if (height > 30) return FontWeight.bold;
    if (height > 24) return FontWeight.w600;
    return FontWeight.normal;
  }
}