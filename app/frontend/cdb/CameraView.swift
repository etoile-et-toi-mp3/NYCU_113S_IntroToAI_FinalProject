//
//  CameraView.swift
//  cdb
//
//  Created by kevin Chou on 2025/4/28.
//

import SwiftUI
import AVFoundation
import PhotosUI

struct CameraView: View {
    @StateObject private var cameraManager = CameraManager()
    @State private var showingPhotoPicker = false
    @State private var capturedImage: UIImage?
    @State private var showingAnalysis = false
    @State private var isCameraReady = false
    
    let onDismiss: () -> Void
    
    var body: some View {
        ZStack {
            // 相機預覽背景
            CameraPreviewView(cameraManager: cameraManager)
                .ignoresSafeArea()
            
            // 頂部狀態欄
            VStack {
                HStack {
                    Button(action: {
                        onDismiss()
                    }) {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 20, weight: .semibold))
                            .foregroundColor(.white)
                            .frame(width: 44, height: 44)
                            .background(Color.black.opacity(0.3))
                            .clipShape(Circle())
                    }
                    
                    Spacer()
                    
                    Text("穿搭分析")
                        .font(.system(size: 18, weight: .bold))
                        .foregroundColor(.white)
                        .shadow(color: .black.opacity(0.5), radius: 2, x: 0, y: 1)
                    
                    Spacer()
                    
                    Button(action: {
                        if isCameraReady {
                            cameraManager.switchCamera()
                        }
                    }) {
                        Image(systemName: "camera.rotate.fill")
                            .font(.system(size: 20))
                            .foregroundColor(.white)
                            .frame(width: 44, height: 44)
                            .background(Color.black.opacity(0.3))
                            .clipShape(Circle())
                    }
                    .disabled(!isCameraReady)
                }
                .padding(.horizontal, 20)
                .padding(.top, 10)
                
                Spacer()
            }
            
            // 底部控制區域
            VStack {
                Spacer()
                
                // 拍照指引
                VStack(spacing: 10) {
                    Text("將穿搭置於畫面中央")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundColor(.white)
                        .shadow(color: .black.opacity(0.7), radius: 2, x: 0, y: 1)
                    
                    Text("確保光線充足，人物完整可見")
                        .font(.system(size: 14))
                        .foregroundColor(.white.opacity(0.8))
                        .shadow(color: .black.opacity(0.7), radius: 2, x: 0, y: 1)
                }
                .padding(.bottom, 30)
                
                // 控制按鈕區域
                HStack(spacing: 40) {
                    // 相簿選擇 - 不依賴相機狀態
                    Button(action: {
                        // 確保沒有其他操作在進行中
                        if !showingAnalysis {
                            showingPhotoPicker = true
                        }
                    }) {
                        VStack(spacing: 8) {
                            ZStack {
                                Circle()
                                    .fill(Color.white.opacity(0.2))
                                    .frame(width: 60, height: 60)
                                
                                Image(systemName: "photo.on.rectangle")
                                    .font(.system(size: 24, weight: .medium))
                                    .foregroundColor(.white)
                            }
                            
                            Text("相簿")
                                .font(.system(size: 12, weight: .medium))
                                .foregroundColor(.white)
                        }
                    }
                    .disabled(showingAnalysis) // 防止重複點擊
                    
                    // 拍照按鈕 - 需要相機準備就緒
                    Button(action: {
                        if isCameraReady && !showingAnalysis {
                            cameraManager.capturePhoto { image in
                                if let image = image {
                                    self.capturedImage = image
                                    // 使用延遲顯示，與相簿選擇保持一致
                                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                                        self.showingAnalysis = true
                                    }
                                }
                            }
                        }
                    }) {
                        ZStack {
                            Circle()
                                .fill(isCameraReady ? Color.white : Color.white.opacity(0.5))
                                .frame(width: 80, height: 80)
                            
                            Circle()
                                .fill(isCameraReady ? Color.white : Color.white.opacity(0.5))
                                .frame(width: 70, height: 70)
                                .overlay(
                                    Circle()
                                        .stroke(Color.black.opacity(0.1), lineWidth: 2)
                                )
                            
                            if !isCameraReady {
                                ProgressView()
                                    .scaleEffect(0.8)
                            }
                        }
                    }
                    .disabled(!isCameraReady || showingAnalysis)
                    .scaleEffect(cameraManager.isCapturing ? 0.9 : 1.0)
                    .animation(.easeInOut(duration: 0.1), value: cameraManager.isCapturing)
                    
                    // 翻轉鏡頭
                    Button(action: {
                        if isCameraReady {
                            cameraManager.switchCamera()
                        }
                    }) {
                        VStack(spacing: 8) {
                            ZStack {
                                Circle()
                                    .fill(Color.white.opacity(0.2))
                                    .frame(width: 60, height: 60)
                                
                                Image(systemName: "arrow.triangle.2.circlepath.camera")
                                    .font(.system(size: 24, weight: .medium))
                                    .foregroundColor(.white)
                            }
                            
                            Text("翻轉")
                                .font(.system(size: 12, weight: .medium))
                                .foregroundColor(.white)
                        }
                    }
                    .disabled(!isCameraReady)
                }
                .padding(.bottom, 50)
            }
            
            // 對焦指示器
            if let focusPoint = cameraManager.focusPoint {
                FocusIndicator()
                    .position(focusPoint)
                    .animation(.easeInOut(duration: 0.3), value: cameraManager.focusPoint)
            }
        }
        .onAppear {
            // 先設置相機準備回調
            cameraManager.onCameraReady = {
                DispatchQueue.main.async {
                    self.isCameraReady = true
                }
            }
            // 請求相機權限並初始化
            cameraManager.requestPermission()
        }
        .onDisappear {
            // 清理回調
            cameraManager.onCameraReady = nil
        }
        .fullScreenCover(isPresented: $showingPhotoPicker) {
            PhotoPicker(selectedImage: $capturedImage) {
                // 延遲顯示分析界面，確保picker完全關閉
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    self.showingAnalysis = true
                }
            }
        }
        .fullScreenCover(isPresented: $showingAnalysis) {
            if let image = capturedImage {
                AnalysisView(selectedImage: image) {
                    // 返回相機時重新啟動相機
                    self.capturedImage = nil
                    self.showingAnalysis = false
                    if isCameraReady {
                        cameraManager.startSession()
                    }
                }
                .onAppear {
                    // 顯示分析視圖時停止相機（如果已啟動）
                    if isCameraReady {
                        cameraManager.stopSession()
                    }
                }
            }
        }
    }
}

// MARK: - 相機預覽視圖
struct CameraPreviewView: UIViewRepresentable {
    let cameraManager: CameraManager
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        
        DispatchQueue.main.async {
            if let previewLayer = cameraManager.previewLayer {
                previewLayer.frame = view.bounds
                previewLayer.videoGravity = .resizeAspectFill
                view.layer.addSublayer(previewLayer)
                
                // 添加手勢識別
                let tapGesture = UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handleTap(_:)))
                view.addGestureRecognizer(tapGesture)
            }
        }
        
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        DispatchQueue.main.async {
            if let previewLayer = cameraManager.previewLayer {
                previewLayer.frame = uiView.bounds
            }
        }
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(cameraManager)
    }
    
    class Coordinator: NSObject {
        let cameraManager: CameraManager
        
        init(_ cameraManager: CameraManager) {
            self.cameraManager = cameraManager
        }
        
        @objc func handleTap(_ gesture: UITapGestureRecognizer) {
            let point = gesture.location(in: gesture.view)
            cameraManager.focus(at: point)
        }
    }
}

// MARK: - 對焦指示器
struct FocusIndicator: View {
    @State private var isAnimating = false
    
    var body: some View {
        ZStack {
            Circle()
                .stroke(Color.yellow, lineWidth: 2)
                .frame(width: 80, height: 80)
                .scaleEffect(isAnimating ? 0.8 : 1.0)
                .opacity(isAnimating ? 0.5 : 1.0)
            
            Circle()
                .stroke(Color.yellow, lineWidth: 1)
                .frame(width: 60, height: 60)
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 0.6).repeatCount(2, autoreverses: true)) {
                isAnimating = true
            }
        }
    }
}

// MARK: - 相簿選擇器
struct PhotoPicker: UIViewControllerRepresentable {
    @Binding var selectedImage: UIImage?
    let onImageSelected: () -> Void
    @Environment(\.presentationMode) var presentationMode
    
    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration()
        config.filter = .images
        config.selectionLimit = 1
        
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = context.coordinator
        
        // 確保有正確的背景和呈現樣式
        picker.view.backgroundColor = UIColor.systemBackground
        picker.modalPresentationStyle = .fullScreen
        
        return picker
    }
    
    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {
        // 確保每次更新時都有正確的背景
        uiViewController.view.backgroundColor = UIColor.systemBackground
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    class Coordinator: NSObject, PHPickerViewControllerDelegate {
        let parent: PhotoPicker
        
        init(_ parent: PhotoPicker) {
            self.parent = parent
        }
        
        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            parent.presentationMode.wrappedValue.dismiss()
            
            // 如果用戶取消選擇（results為空），直接返回
            guard let provider = results.first?.itemProvider else { 
                return 
            }
            
            if provider.canLoadObject(ofClass: UIImage.self) {
                provider.loadObject(ofClass: UIImage.self) { image, _ in
                    DispatchQueue.main.async {
                        self.parent.selectedImage = image as? UIImage
                        self.parent.onImageSelected()
                    }
                }
            }
        }
    }
}

// MARK: - 相機管理器
class CameraManager: NSObject, ObservableObject {
    @Published var isCapturing = false
    @Published var focusPoint: CGPoint?
    
    private let captureSession = AVCaptureSession()
    private var videoDeviceInput: AVCaptureDeviceInput?
    private let photoOutput = AVCapturePhotoOutput()
    
    lazy var previewLayer: AVCaptureVideoPreviewLayer? = {
        let layer = AVCaptureVideoPreviewLayer(session: captureSession)
        return layer
    }()
    
    private var photoCaptureCompletion: ((UIImage?) -> Void)?
    
    var onCameraReady: (() -> Void)?
    
    override init() {
        super.init()
        setupCamera()
    }
    
    func requestPermission() {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            DispatchQueue.main.async {
                if granted {
                    self.startSession()
                    // 相機會話啟動完成後，通知界面相機已準備就緒
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        self.onCameraReady?()
                    }
                } else {
                    // 即使沒有相機權限，也要通知準備完成（用戶仍可使用相簿功能）
                    self.onCameraReady?()
                }
            }
        }
    }
    
    private func setupCamera() {
        captureSession.sessionPreset = .photo
        
        guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let videoDeviceInput = try? AVCaptureDeviceInput(device: videoDevice) else {
            // 即使相機設置失敗，也通知準備完成
            DispatchQueue.main.async {
                self.onCameraReady?()
            }
            return
        }
        
        if captureSession.canAddInput(videoDeviceInput) {
            captureSession.addInput(videoDeviceInput)
            self.videoDeviceInput = videoDeviceInput
        }
        
        if captureSession.canAddOutput(photoOutput) {
            captureSession.addOutput(photoOutput)
        }
    }
    
    func startSession() {
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
        }
    }
    
    func stopSession() {
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.stopRunning()
        }
    }
    
    func switchCamera() {
        guard let currentInput = videoDeviceInput else { return }
        
        let currentPosition = currentInput.device.position
        let newPosition: AVCaptureDevice.Position = currentPosition == .back ? .front : .back
        
        guard let newDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: newPosition),
              let newInput = try? AVCaptureDeviceInput(device: newDevice) else {
            return
        }
        
        captureSession.beginConfiguration()
        captureSession.removeInput(currentInput)
        
        if captureSession.canAddInput(newInput) {
            captureSession.addInput(newInput)
            videoDeviceInput = newInput
        } else {
            captureSession.addInput(currentInput)
        }
        
        captureSession.commitConfiguration()
    }
    
    func focus(at point: CGPoint) {
        guard let device = videoDeviceInput?.device,
              device.isFocusPointOfInterestSupported else { return }
        
        // 轉換座標
        guard let previewLayer = previewLayer else { return }
        let focusPoint = previewLayer.captureDevicePointConverted(fromLayerPoint: point)
        
        do {
            try device.lockForConfiguration()
            device.focusPointOfInterest = focusPoint
            device.focusMode = .autoFocus
            device.unlockForConfiguration()
            
            DispatchQueue.main.async {
                self.focusPoint = point
                
                // 2秒後隱藏對焦指示器
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                    self.focusPoint = nil
                }
            }
        } catch {
            print("無法設置對焦: \(error)")
        }
    }
    
    func capturePhoto(completion: @escaping (UIImage?) -> Void) {
        isCapturing = true
        photoCaptureCompletion = completion
        
        // 根據支持的編碼格式創建設置
        let settings: AVCapturePhotoSettings
        if photoOutput.availablePhotoCodecTypes.contains(.hevc) {
            settings = AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.hevc])
        } else {
            settings = AVCapturePhotoSettings()
        }
        
        photoOutput.capturePhoto(with: settings, delegate: self)
    }
}

// MARK: - 相機拍照代理
extension CameraManager: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        isCapturing = false
        
        guard let imageData = photo.fileDataRepresentation(),
              let image = UIImage(data: imageData) else {
            photoCaptureCompletion?(nil)
            return
        }
        
        photoCaptureCompletion?(image)
        photoCaptureCompletion = nil
    }
} 