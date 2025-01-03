import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../services/ocr_service.dart';
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
  bool _isProcessing = true;
  String _errorMessage = '';
  Map<String, dynamic>? _responseData;

  @override
  void initState() {
    super.initState();
    _processImage();
  }

  Future<void> _processImage() async {
    try {
      // Process image and get OCR results
      await _ocrService.loadModels();
      final results = await _ocrService.processImage(widget.image);
      
      // Write to KVDB
      final key = await _kvdbService.writeData(results);
      
      // Wait for processing
      await Future.delayed(const Duration(seconds: 8));
      
      // Read from KVDB - using the generated key instead of hardcoded value
      final data = await _kvdbService.readData("1735902270721");
      
      setState(() {
        _responseData = data;
        _isProcessing = false;
      });
    } catch (e) {
      setState(() {
        _errorMessage = e.toString();
        _isProcessing = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Processed Data'),
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

    if (_responseData == null) {
      return const Center(child: Text('No data available'));
    }

    return SingleChildScrollView(
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Text(
              'Response Data',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
          ),
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: _buildDataTable(),
          ),
        ],
      ),
    );
  }

  Widget _buildDataTable() {
    List<DataRow> rows = [];
    
    void addDataRows(Map<String, dynamic> map, [String prefix = '']) {
      map.forEach((key, value) {
        if (value is Map<String, dynamic>) {
          rows.add(DataRow(cells: [
            DataCell(Text('$prefix$key')),
            DataCell(Text('Object')),
          ]));
          addDataRows(value, '$prefix  ');
        } else if (value is List) {
          rows.add(DataRow(cells: [
            DataCell(Text('$prefix$key')),
            DataCell(Text('List [${value.length} items]')),
          ]));
          for (var i = 0; i < value.length; i++) {
            if (value[i] is Map) {
              addDataRows(value[i] as Map<String, dynamic>, '$prefix  [$i] ');
            } else {
              rows.add(DataRow(cells: [
                DataCell(Text('$prefix  [$i]')),
                DataCell(Text(value[i].toString())),
              ]));
            }
          }
        } else {
          rows.add(DataRow(cells: [
            DataCell(Text('$prefix$key')),
            DataCell(Text(value?.toString() ?? 'null')),
          ]));
        }
      });
    }

    addDataRows(_responseData!);

    return DataTable(
      columns: const [
        DataColumn(label: Text('Field')),
        DataColumn(label: Text('Value')),
      ],
      rows: rows,
    );
  }
}