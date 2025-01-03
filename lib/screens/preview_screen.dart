import 'dart:ui' as ui;
import 'dart:io'; 
import 'dart:convert'; 
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:json_table/json_table.dart';
import 'package:http/http.dart' as http;
import '../services/ocr_service.dart';
import '../models/bounding_box.dart';
import '../services/bounding_box_painter.dart';
import '../models/ocr_result.dart';
import '../services/kvdb_service.dart';

class PreviewScreen extends StatefulWidget {
  final XFile image;

  const PreviewScreen({Key? key, required this.image}) : super(key: key);

  @override
  _PreviewScreenState createState() => _PreviewScreenState();
}

class _PreviewScreenState extends State<PreviewScreen> {
  final OCRService _ocrService = OCRService();
  final KVDBService _kvdbService = KVDBService();
  List<OCRResult> _results = [];
  bool _isProcessing = true;
  String _errorMessage = '';
  String? _kvdbKey;
  Map<String, dynamic>? _kvdbData;

  @override
  void initState() {
    super.initState();
    _processImage();
  }

   Future<void> _processImage() async {
    try {
      await _ocrService.loadModels();
      final results = await _ocrService.processImage(widget.image);
      
      // Write to KVDB
      final key = await _kvdbService.writeData(results).catchError((error) {
        throw Exception('Failed to save data: ${error.toString()}');
      });
      
      setState(() {
        _results = results;
        _kvdbKey = key;
        _isProcessing = false;
      });
      
      // Wait for 8 seconds before reading back from KVDB
      await Future.delayed(const Duration(seconds: 8));
      
      // Read from KVDB for formatted
      final data = await _kvdbService.readData("1735902270721").catchError((error) {
        throw Exception('Failed to fetch data: ${error.toString()}');
      });

      setState(() {
        _kvdbData = data;
      });
    } catch (e) {
      setState(() {
        _errorMessage = 'Error: ${e.toString()}';
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

    return SingleChildScrollView(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (_kvdbKey != null) ...[
              Text('KVDB Key: $_kvdbKey', 
                style: const TextStyle(fontWeight: FontWeight.bold)),
              const SizedBox(height: 16),
            ],
            
            const SizedBox(height: 16),
            if (_kvdbData == null) ...[
              const Center(
                child: Column(
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 8),
                    Text('Crunching the data...'),
                  ],
                ),
              ),
            ] else ...[
              const Text('Extracted Data:', 
                style: TextStyle(fontWeight: FontWeight.bold)),
              const SizedBox(height: 8),
              SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                child: JsonTable(
                  _kvdbData?['Processed_data'] as List?
                  showColumnToggle: true,
                  tableCellBuilder: (value) {
                    return Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 8.0, 
                        vertical: 4.0
                      ),
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.grey.shade300),
                      ),
                      child: Text(
                        value.toString(),
                        style: const TextStyle(fontSize: 14),
                      ),
                    );
                  },
                ),
              ),
            ],
          ],
        ),
      ),
    );
}

}