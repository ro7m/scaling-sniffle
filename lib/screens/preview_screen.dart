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

    return ListView.builder(
      padding: const EdgeInsets.all(16.0),
      itemCount: _results.length,
      itemBuilder: (context, index) {
        final result = _results[index];
        return Card(
          child: Padding(
            padding: const EdgeInsets.all(8.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  result.text,
                  style: const TextStyle(fontSize: 16.0),
                )
              ],
            ),
          ),
        );
      },
    );
  }
}