import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';
import 'package:path_provider/path_provider.dart';

class PreviewScreen extends StatefulWidget {
  final String? msgkey;

  const PreviewScreen({Key? key, required this.msgkey}) : super(key: key);

  @override
  _PreviewScreenState createState() => _PreviewScreenState();
}

class _PreviewScreenState extends State<PreviewScreen> {
  late Future<Map<String, dynamic>> _futureData;

  @override
  void initState() {
    super.initState();
    if (widget.msgkey != '') {
      _futureData = _fetchData();
    } else {
      _futureData = Future.value({});
    }
  }

  Future<Map<String, dynamic>> _fetchData() async {
    try {
      final response = await http.get(
        Uri.parse('https://kvdb.io/VuKUzo8aFSpoWpyXKpFxxH/${widget.msgkey}'),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Failed to read data from upstream');
      }
    } catch (e) {
      print('Exception occurred while fetching data for msgkey: ${widget.msgkey}');
      print(e);
      rethrow;
    }
  }

  Future<void> _downloadCsv(Map<String, dynamic> data) async {
    final flattenedData = _flattenJson(data);
    final keys = flattenedData.keys.toList();
    final values = keys.map((key) => flattenedData[key]).toList();

    final String csvData = '${keys.join(',')}\n${values.join(',')}';
    final directory = await getDownloadsDirectory();
    final path = '${directory!.path}/data.csv';
    final File file = File(path);
    await file.writeAsString(csvData);

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('CSV downloaded to $path')),
    );
  }

  Map<String, dynamic> _flattenJson(Map<String, dynamic> json) {
    Map<String, dynamic> flattened = {};

    void _extract(String prefix, dynamic value) {
      if (value is Map<String, dynamic>) {
        value.forEach((key, val) {
          _extract('$prefix.$key', val);
        });
      } else if (value is List) {
        for (int i = 0; i < value.length; i++) {
          _extract('$prefix[$i]', value[i]);
        }
      } else {
        flattened[prefix] = value.toString();
      }
    }

    json.forEach((key, value) {
      _extract(key, value);
    });

    return flattened;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Extracted Data'),
      ),
      body: FutureBuilder<Map<String, dynamic>>(
        future: _futureData,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return Center(child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: const [
                CircularProgressIndicator(),
                SizedBox(height: 20),
                Text('Fetching processed data...')
              ],
            ));
          } else if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          } else if (!snapshot.hasData || snapshot.data == null) {
            return Center(child: Text('No data found'));
          }

          final data = snapshot.data!;
          final flattenedData = _flattenJson(data);
          final keys = flattenedData.keys.toList();
          final values = [flattenedData.values.toList()];

          return Column(
            children: [
              Expanded(
                child: SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: SingleChildScrollView(
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: DataTable(
                        headingTextStyle: const TextStyle(
                          fontWeight: FontWeight.bold,
                          color: Colors.blue,
                        ),
                        columns: keys.map((key) => DataColumn(label: Text(key))).toList(),
                        rows: values.map((row) {
                          return DataRow(
                            cells: row.map((value) => DataCell(Text(value.toString()))).toList(),
                          );
                        }).toList(),
                      ),
                    ),
                  ),
                ),
              ),
              SizedBox(
                height: 50.0,
                child: ElevatedButton(
                  onPressed: () => _downloadCsv(data),
                  child: Text('Download as CSV'),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}