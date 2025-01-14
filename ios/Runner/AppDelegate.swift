import Flutter
import UIKit

@main
@objc class AppDelegate: FlutterAppDelegate {
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    GeneratedPluginRegistrant.register(with: self)
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  private func setupMethodChannel(_ controller: FlutterViewController) {
    let channel = FlutterMethodChannel(
        name: "flutter_device_type",
        binaryMessenger: controller.binaryMessenger)
    
    channel.setMethodCallHandler { call, result in
        if call.method == "isRealDevice" {
            #if targetEnvironment(simulator)
            result("false")
            #else
            result("true")
            #endif
        } else {
            result(FlutterMethodNotImplemented)
        }
    }
}

}
