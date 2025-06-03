//
//  SettingsManager.swift
//  cdb
//
//  Created by AI Assistant on 2025/1/1.
//

import SwiftUI
import Foundation

// MARK: - 設定管理器
class SettingsManager: ObservableObject {
    static let shared = SettingsManager()
    
    @Published var baseURL: String {
        didSet {
            UserDefaults.standard.set(baseURL, forKey: "baseURL")
        }
    }
    
    private init() {
        // 從 UserDefaults 讀取儲存的 URL，如果沒有則使用預設值
        self.baseURL = UserDefaults.standard.string(forKey: "baseURL") ?? "http://192.168.0.247:8000"
    }
    
    // 重置為預設值
    func resetToDefault() {
        baseURL = "http://192.168.0.247:8000"
    }
    
    // 驗證 URL 格式
    func isValidURL(_ urlString: String) -> Bool {
        guard let url = URL(string: urlString) else { return false }
        return url.scheme != nil && url.host != nil
    }
}

// MARK: - 設定頁面
struct SettingsView: View {
    @ObservedObject var settingsManager = SettingsManager.shared
    @Environment(\.presentationMode) var presentationMode
    @State private var tempURL: String = ""
    @State private var showingAlert = false
    @State private var alertMessage = ""
    @State private var isTestingConnection = false
    @State private var testResult = ""
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("API 設定")) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("FastAPI 伺服器地址")
                            .font(.system(size: 16, weight: .medium))
                            .foregroundColor(.black)
                        
                        TextField("http://192.168.0.247:8000", text: $tempURL)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .autocapitalization(.none)
                            .disableAutocorrection(true)
                            .keyboardType(.URL)
                        
                        Text("請輸入完整的 URL，包含 http:// 或 https://")
                            .font(.system(size: 12))
                            .foregroundColor(.gray)
                    }
                    .padding(.vertical, 8)
                    
                    HStack {
                        Button("測試連線") {
                            testConnection()
                        }
                        .disabled(tempURL.isEmpty || isTestingConnection)
                        
                        Spacer()
                        
                        if isTestingConnection {
                            ProgressView()
                                .scaleEffect(0.8)
                        }
                    }
                    
                    if !testResult.isEmpty {
                        Text(testResult)
                            .font(.system(size: 12))
                            .foregroundColor(testResult.contains("成功") ? .green : .red)
                    }
                }
                
                Section(header: Text("快速設定")) {
                    Button("使用預設值") {
                        tempURL = "http://192.168.0.247:8000"
                    }
                    
                    Button("本地開發環境") {
                        tempURL = "http://localhost:8000"
                    }
                    
                    Button("本地網路（127.0.0.1）") {
                        tempURL = "http://127.0.0.1:8000"
                    }
                }
                
                Section(header: Text("關於")) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("當前設定")
                            .font(.system(size: 14, weight: .medium))
                        Text(settingsManager.baseURL)
                            .font(.system(size: 12, weight: .regular))
                            .foregroundColor(.gray)
                            .textSelection(.enabled)
                    }
                    .padding(.vertical, 4)
                }
            }
            .navigationTitle("應用程式設定")
            .navigationBarItems(
                leading: Button("取消") {
                    presentationMode.wrappedValue.dismiss()
                },
                trailing: Button("儲存") {
                    saveSettings()
                }
                .disabled(tempURL.isEmpty)
            )
        }
        .onAppear {
            tempURL = settingsManager.baseURL
        }
        .alert(isPresented: $showingAlert) {
            Alert(
                title: Text("設定"),
                message: Text(alertMessage),
                dismissButton: .default(Text("確定"))
            )
        }
    }
    
    private func saveSettings() {
        guard !tempURL.isEmpty else {
            alertMessage = "請輸入有效的 URL"
            showingAlert = true
            return
        }
        
        guard settingsManager.isValidURL(tempURL) else {
            alertMessage = "請輸入有效的 URL 格式，例如：http://192.168.0.247:8000"
            showingAlert = true
            return
        }
        
        settingsManager.baseURL = tempURL
        alertMessage = "設定已儲存！重新啟動應用程式後生效。"
        showingAlert = true
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            presentationMode.wrappedValue.dismiss()
        }
    }
    
    private func testConnection() {
        guard !tempURL.isEmpty else {
            testResult = "❌ 請先輸入 URL"
            return
        }
        
        guard settingsManager.isValidURL(tempURL) else {
            testResult = "❌ URL 格式不正確"
            return
        }
        
        isTestingConnection = true
        testResult = "測試中..."
        
        // 簡單的連線測試
        Task {
            do {
                let url = URL(string: "\(tempURL)/")!
                let (_, response) = try await URLSession.shared.data(from: url)
                
                await MainActor.run {
                    if let httpResponse = response as? HTTPURLResponse {
                        if httpResponse.statusCode == 200 {
                            testResult = "✅ 連線成功"
                        } else {
                            testResult = "⚠️ 伺服器回應碼: \(httpResponse.statusCode)"
                        }
                    } else {
                        testResult = "⚠️ 收到回應但格式異常"
                    }
                    isTestingConnection = false
                }
            } catch {
                await MainActor.run {
                    testResult = "❌ 連線失敗: \(error.localizedDescription)"
                    isTestingConnection = false
                }
            }
        }
    }
}

struct SettingsView_Previews: PreviewProvider {
    static var previews: some View {
        SettingsView()
    }
} 