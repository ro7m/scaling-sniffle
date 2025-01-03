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
      
      // Read from KVDB using the generated key
      final data = await _kvdbService.readData("1735902270721").catchError((error) {
        throw Exception('Failed to fetch data: ${error.toString()}');
      });

      print('KVDB Data received: ${jsonEncode(data)}'); // Debug print

      setState(() {
        _kvdbData = data;
      });
    } catch (e) {
      print('Error in _processImage: $e'); // Debug print
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
            // Debug Information Section
            Container(
              padding: const EdgeInsets.all(8.0),
              color: Colors.grey[200],
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('DEBUG INFORMATION',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      color: Colors.red
                    )
                  ),
                  const SizedBox(height: 8),
                  Text('KVDB Data: ${_kvdbData.toString()}',
                    style: const TextStyle(
                      fontSize: 12,
                      fontFamily: 'Courier',
                    )
                  ),
                  const SizedBox(height: 8),
                  Text('KVDB Data Type: ${_kvdbData?.runtimeType}',
                    style: const TextStyle(
                      fontSize: 12,
                      fontFamily: 'Courier',
                    )
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),

            // Original Results Section
            const Text('Original Results:', 
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18)),
            const SizedBox(height: 8),
            ..._results.map((result) => Padding(
              padding: const EdgeInsets.only(bottom: 8.0),
              child: Text(result.text),
            )).toList(),
            
            const SizedBox(height: 24),
            
            if (_kvdbData != null) ...[
              const Text('KVDB Data:', 
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18)),
              const SizedBox(height: 8),
              _buildKVDBDataTable(),
            ] else ...[
              const Center(
                child: Column(
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 8),
                    Text('Loading KVDB data...'),
                  ],
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildKVDBDataTable() {
    if (_kvdbData == null) {
      return const Text('No data available');
    }

    // Convert the data to a format that JsonTable can understand
    final List<Map<String, dynamic>> tableData = [_kvdbData!];

    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          JsonTable(
            tableData,
            showColumnToggle: true,
            allowRowHighlight: true,
            rowHighlightColor: Colors.yellow[500]!.withOpacity(0.7),
            tableHeaderBuilder: (String? header) {
              return Container(
                padding: const EdgeInsets.symmetric(horizontal: 8.0, vertical: 4.0),
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey.shade300),
                  color: Colors.grey.shade100,
                ),
                child: Text(
                  (header ?? '').toUpperCase(),
                  style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 15),
                ),
              );
            },
            tableCellBuilder: (value) {
              return Container(
                padding: const EdgeInsets.symmetric(horizontal: 8.0, vertical: 4.0),
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey.shade300),
                ),
                child: Text(
                  value?.toString() ?? 'N/A',
                  style: const TextStyle(fontSize: 14),
                ),
              );
            },
          ),
        ],
      ),
    );
  }
}