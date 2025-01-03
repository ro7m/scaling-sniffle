import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:device_info_plus/device_info_plus.dart';
import 'dart:io' show Platform;
import '../models/ocr_result.dart';

class KVDBService {
  static const String writeUrl = 'https://kvdb.io/NyKpFtJ7v392NS8ibLiofx';
  static const String readUrl = 'https://kvdb.io/VuKUzo8aFSpoWpyXKpFxxH';
  
  Future<String> writeData(List<OCRResult> results) async {
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final deviceInfo = await _getDeviceInfo();
    
    final data = {
      'extractedAt': timestamp,
      'probableTextContent': results.map((r) => r.text).join(' '),
      'boundingBoxes': results.map((r) => {
        'text': r.text,
        'x': r.boundingBox.x,
        'y': r.boundingBox.y,
        'width': r.boundingBox.width,
        'height': r.boundingBox.height,
      }).toList(),
      'deviceInfo': deviceInfo,
    };

    final response = await http.put(
      Uri.parse('$writeUrl/$timestamp'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(data),
    );

    if (response.statusCode != 200) {
      throw Exception('Failed to write to KVDB: ${response.statusCode}');
    }

    return timestamp.toString();
  }

  Future<Map<String, dynamic>> readData(String key) async {
    final response = await http.get(Uri.parse('$readUrl/$key'));
    
    if (response.statusCode != 200) {
      throw Exception('Failed to read from KVDB: ${response.statusCode}');
    }

    return jsonDecode(response.body);
  }

  Future<Map<String, dynamic>> _getDeviceInfo() async {
    final deviceInfo = DeviceInfoPlugin();
    if (Platform.isAndroid) {
      final androidInfo = await deviceInfo.androidInfo;
      return {
        'platform': 'Android',
        'device': androidInfo.model,
        'manufacturer': androidInfo.manufacturer,
      };
    } else if (Platform.isIOS) {
      final iosInfo = await deviceInfo.iosInfo;
      return {
        'platform': 'iOS',
        'device': iosInfo.model,
        'systemVersion': iosInfo.systemVersion,
      };
    }
    return {'platform': 'Unknown'};
  }
}