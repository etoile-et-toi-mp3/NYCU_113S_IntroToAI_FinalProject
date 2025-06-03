//
//  FeedView.swift
//  cdb
//
//  Created by kevin Chou on 2025/4/28.
//

import SwiftUI
import PhotosUI

// MARK: - 數據模型
struct FashionRecommendation: Identifiable, Codable {
    let id = UUID()
    let recommendationId: String
    let path: String
    let style: String
    let gender: String
    let similarity: Double
    let score: Double
    let detailedSimilarity: DetailedSimilarity
    
    enum CodingKeys: String, CodingKey {
        case recommendationId = "recommendation_id"
        case path, style, gender, similarity, score
        case detailedSimilarity = "detailed_similarity"
    }
}

struct DetailedSimilarity: Codable {
    let visualSimilarity: Double
    let mainComponentSimilarity: Double
    let styleSimilarity: Double?
    
    enum CodingKeys: String, CodingKey {
        case visualSimilarity = "visual_similarity"
        case mainComponentSimilarity = "main_component_similarity"
        case styleSimilarity = "style_similarity"
    }
}

struct RecommendationResponse: Codable {
    let status: String
    let requestId: String
    let inputImageUrl: String
    let analysisTime: Double
    let recommendations: [FashionRecommendation]
    let styleAnalysis: StyleAnalysis
    
    enum CodingKeys: String, CodingKey {
        case status
        case requestId = "request_id"
        case inputImageUrl = "input_image_url"
        case analysisTime = "analysis_time"
        case recommendations
        case styleAnalysis = "style_analysis"
    }
}

struct StyleAnalysis: Codable {
    let dominantStyle: String
    let averageVisualSimilarity: Double
    let styleDistribution: [String: Int]
    
    enum CodingKeys: String, CodingKey {
        case dominantStyle = "dominant_style"
        case averageVisualSimilarity = "average_visual_similarity"
        case styleDistribution = "style_distribution"
    }
}

struct AdviceResponse: Codable {
    let status: String
    let requestId: String
    let recommendationId: String
    let targetStyle: String
    let aiAdvice: [String: String]
    let analysisTime: Double
    
    enum CodingKeys: String, CodingKey {
        case status
        case requestId = "request_id"
        case recommendationId = "recommendation_id"
        case targetStyle = "target_style"
        case aiAdvice = "ai_advice"
        case analysisTime = "analysis_time"
    }
}

// MARK: - API服務
class FashionAPIService: ObservableObject {
    @ObservedObject private var settingsManager = SettingsManager.shared
    
    private var baseURL: String {
        return settingsManager.baseURL
    }
    
    func getRecommendations(image: UIImage, gender: String, stylePreference: String? = nil, strategy: String = "balanced") async throws -> RecommendationResponse {
        guard let url = URL(string: "\(baseURL)/recommend") else {
            throw APIError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        
        // 添加圖片
        if let imageData = image.jpegData(compressionQuality: 0.8) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"image\"; filename=\"photo.jpg\"\r\n".data(using: .utf8)!)
            body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
            body.append(imageData)
            body.append("\r\n".data(using: .utf8)!)
        }
        
        // 添加其他參數
        let parameters = [
            "gender": gender,
            "style_preference": stylePreference ?? "",
            "top_k": "4",
            "strategy": strategy
        ]
        
        for (key, value) in parameters {
            if !value.isEmpty {
                body.append("--\(boundary)\r\n".data(using: .utf8)!)
                body.append("Content-Disposition: form-data; name=\"\(key)\"\r\n\r\n".data(using: .utf8)!)
                body.append("\(value)\r\n".data(using: .utf8)!)
            }
        }
        
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        request.httpBody = body
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw APIError.serverError
        }
        
        let recommendationResponse = try JSONDecoder().decode(RecommendationResponse.self, from: data)
        return recommendationResponse
    }
    
    func getAdvice(userImagePath: String, targetImagePath: String, targetStyle: String, recommendationId: String, aiModels: [String] = ["rule_based", "clip"]) async throws -> AdviceResponse {
        guard let url = URL(string: "\(baseURL)/advice") else {
            throw APIError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        
        // 根據是否使用LLaVA設置超時時間
        let hasLLaVA = aiModels.contains("llava")
        request.timeoutInterval = hasLLaVA ? 300.0 : 60.0 // LLaVA: 5分鐘，其他: 1分鐘
        
        let parameters = [
            "user_image_path": userImagePath,
            "target_image_path": targetImagePath,
            "target_style": targetStyle,
            "ai_models": aiModels.joined(separator: ","),
            "recommendation_id": recommendationId
        ]
        
        let bodyString = parameters.map { "\($0.key)=\($0.value.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? "")" }.joined(separator: "&")
        request.httpBody = bodyString.data(using: .utf8)
        
        // 創建自定義URLSession配置以支持更長超時
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = hasLLaVA ? 300.0 : 60.0
        config.timeoutIntervalForResource = hasLLaVA ? 300.0 : 60.0
        let session = URLSession(configuration: config)
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.serverError
        }
        
        // 改善錯誤處理
        if httpResponse.statusCode != 200 {
            if hasLLaVA && (httpResponse.statusCode == 408 || httpResponse.statusCode == 504) {
                throw APIError.llavaTimeout
            } else {
                throw APIError.serverError
            }
        }
        
        do {
            let adviceResponse = try JSONDecoder().decode(AdviceResponse.self, from: data)
            return adviceResponse
        } catch {
            print("❌ JSON解碼錯誤: \(error)")
            print("🔍 服務器回應: \(String(data: data, encoding: .utf8) ?? "無法解碼")")
            throw APIError.decodingError
        }
    }
    
    // 添加圖片URL生成方法
    func getImageURL(path: String) -> URL? {
        print("🔍 處理圖片路徑: \(path)")
        
        // 如果是絕對URL，直接使用
        if path.hasPrefix("http") {
            print("✅ 使用絕對URL: \(path)")
            return URL(string: path)
        }
        
        // 處理相對路徑，構建完整URL
        let cleanPath = path.hasPrefix("/") ? String(path.dropFirst()) : path
        let fullURL = "\(baseURL)/\(cleanPath)"
        print("✅ 構建完整URL: \(fullURL)")
        
        return URL(string: fullURL)
    }
}

enum APIError: Error {
    case invalidURL
    case serverError
    case decodingError
    case llavaTimeout
}

// MARK: - 主視圖
struct FeedView: View {
    @StateObject private var apiService = FashionAPIService()
    @State private var selectedImage: UIImage?
    @State private var showingImagePicker = false
    @State private var showingCameraOptions = false
    @State private var selectedGender = "MEN"
    @State private var selectedStyle = ""
    @State private var selectedStrategy = "balanced"
    @State private var isAnalyzing = false
    @State private var recommendations: [FashionRecommendation] = []
    @State private var styleAnalysis: StyleAnalysis?
    @State private var errorMessage = ""
    @State private var showingAdvice = false
    @State private var selectedRecommendation: FashionRecommendation?
    @State private var adviceData: AdviceResponse?
    @State private var userImagePath = ""
    @State private var selectedAIModels: Set<String> = ["rule_based", "clip"]
    @State private var isGeneratingAdvice = false
    @State private var generatingAdviceForRecommendation: Set<String> = []
    @State private var completedAdviceRecommendations: Set<String> = []
    @State private var imageSourceType: UIImagePickerController.SourceType = .photoLibrary
    
    let genders = ["MEN", "WOMEN"]
    let styles = ["", "CASUAL", "STREET", "FORMAL", "BOHEMIAN"]
    let strategies = [
        ("balanced", "平衡推薦"),
        ("pure_visual", "視覺優先"),
        ("style_aware", "風格導向")
    ]
    
    let aiModels = [
        ("rule_based", "規則系統", "快速", true),
        ("clip", "FashionCLIP", "詳細特徵分析", true),
        ("llava", "視覺語言模型", "深度分析", false)
    ]
    
    var body: some View {
        NavigationView {
                ScrollView {
                    VStack(spacing: 25) {
                    // 標題區域
                    VStack(spacing: 10) {
                        Text("穿搭分析")
                            .font(.system(size: 28, weight: .bold, design: .rounded))
                            .foregroundColor(.black)
                        
                        Text("上傳你的穿搭照片，獲得AI智能推薦")
                            .font(.system(size: 16, weight: .medium))
                            .foregroundColor(.gray)
                    }
                    .padding(.top, 20)
                    
                    // 圖片上傳區域
                    uploadImageSection
                    
                    // 設置區域
                    if selectedImage != nil {
                        settingsSection
                    }
                    
                    // 分析按鈕
                    if selectedImage != nil && !isAnalyzing {
                        analyzeButton
                    }
                    
                    // 加載指示器
                    if isAnalyzing {
                        loadingView
                    }
                    
                    // 錯誤信息
                    if !errorMessage.isEmpty {
                        errorView
                    }
                    
                    // 推薦結果
                    if !recommendations.isEmpty {
                        recommendationsSection
                    }
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 30)
            }
            .background(Color.white.ignoresSafeArea())
            .navigationBarTitle("穿搭分析", displayMode: .inline)
            .sheet(isPresented: $showingImagePicker) {
                ImagePickerView(selectedImage: $selectedImage, sourceType: imageSourceType)
                        }
            .sheet(isPresented: $showingAdvice) {
                if let advice = adviceData, let recommendation = selectedRecommendation {
                    AdviceDetailView(advice: advice, recommendation: recommendation)
                    }
            }
            .actionSheet(isPresented: $showingCameraOptions) {
                ActionSheet(
                    title: Text("選擇穿搭照片"),
                    buttons: [
                        .default(Text("拍照")) {
                            imageSourceType = .camera
                            selectedImage = nil // 重置選中圖片
                            showingImagePicker = true
                        },
                        .default(Text("從相簿選擇")) {
                            imageSourceType = .photoLibrary
                            selectedImage = nil // 重置選中圖片
                            showingImagePicker = true
                        },
                        .cancel(Text("取消"))
                    ]
                )
            }
        }
    }
    
    // MARK: - 子視圖
    private var uploadImageSection: some View {
        Button(action: {
            showingCameraOptions = true
        }) {
            ZStack {
                RoundedRectangle(cornerRadius: 20)
                    .fill(Color.gray.opacity(0.1))
                    .frame(height: 300)
                    .overlay(
                        RoundedRectangle(cornerRadius: 20)
                            .stroke(Color.gray.opacity(0.3), style: StrokeStyle(lineWidth: 2, dash: [10]))
                    )
                
                if let image = selectedImage {
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: UIScreen.main.bounds.width - 40, height: 300)
                        .clipped()
                        .cornerRadius(20)
                } else {
                    VStack(spacing: 15) {
                Image(systemName: "camera.fill")
                            .font(.system(size: 40))
                            .foregroundColor(.gray)
                        
                        Text("點擊上傳穿搭照片")
                    .font(.system(size: 18, weight: .medium))
                            .foregroundColor(.gray)
                
                        Text("支持拍照或從相簿選擇")
                            .font(.system(size: 14))
                            .foregroundColor(.gray.opacity(0.8))
                        
                        Text("📌 建議：穿搭照片效果更佳")
                            .font(.system(size: 12))
                            .foregroundColor(.blue.opacity(0.8))
        }
    }
            }
        }
    }
    
    private var settingsSection: some View {
        VStack(spacing: 20) {
            // 性別選擇
            VStack(alignment: .leading, spacing: 10) {
                Text("性別")
                    .font(.system(size: 16, weight: .medium))
                .foregroundColor(.black)
                
                HStack {
                    ForEach(genders, id: \.self) { gender in
                        Button(action: {
                            selectedGender = gender
                        }) {
                            Text(gender == "MEN" ? "男性" : "女性")
                                .font(.system(size: 14, weight: .medium))
                                .foregroundColor(selectedGender == gender ? .white : .black)
                    .padding(.horizontal, 20)
                                .padding(.vertical, 10)
                                .background(
                                    RoundedRectangle(cornerRadius: 20)
                                        .fill(selectedGender == gender ? Color.black : Color.gray.opacity(0.1))
                                )
                        }
                    }
                    Spacer()
                }
            }
            
            // 風格偏好
            VStack(alignment: .leading, spacing: 10) {
                Text("風格偏好 (可選)")
                    .font(.system(size: 16, weight: .medium))
                .foregroundColor(.black)
                
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack {
                        ForEach(styles, id: \.self) { style in
                            Button(action: {
                                selectedStyle = style
                            }) {
                                Text(styleDisplayName(style))
                                    .font(.system(size: 14, weight: .medium))
                                    .foregroundColor(selectedStyle == style ? .white : .black)
                                    .padding(.horizontal, 16)
                                    .padding(.vertical, 8)
                                    .background(
                                        RoundedRectangle(cornerRadius: 16)
                                            .fill(selectedStyle == style ? Color.black : Color.gray.opacity(0.1))
                                    )
                            }
                        }
                    }
                    .padding(.horizontal, 5)
                }
            }
            
            // 推薦策略
            VStack(alignment: .leading, spacing: 10) {
                Text("推薦策略")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(.black)
                
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack {
                        ForEach(strategies, id: \.0) { strategy in
                            Button(action: {
                                selectedStrategy = strategy.0
                            }) {
                                Text(strategy.1)
                                    .font(.system(size: 14, weight: .medium))
                                    .foregroundColor(selectedStrategy == strategy.0 ? .white : .black)
                                    .padding(.horizontal, 16)
                                    .padding(.vertical, 8)
                                    .background(
                                        RoundedRectangle(cornerRadius: 16)
                                            .fill(selectedStrategy == strategy.0 ? Color.black : Color.gray.opacity(0.1))
                                    )
                            }
                        }
                    }
                    .padding(.horizontal, 5)
        }
        }
            
            // AI模型選擇
            VStack(alignment: .leading, spacing: 10) {
                Text("AI建議模型")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(.black)
                
                Text("選擇想要使用的AI模型來生成穿搭建議")
                    .font(.system(size: 12))
                    .foregroundColor(.gray)
                
                VStack(spacing: 8) {
                    ForEach(aiModels, id: \.0) { model in
                        let (id, name, description, _) = model
                        
                        Button(action: {
                            if selectedAIModels.contains(id) {
                                selectedAIModels.remove(id)
                            } else {
                                selectedAIModels.insert(id)
                            }
                        }) {
            HStack {
                                Image(systemName: selectedAIModels.contains(id) ? "checkmark.square.fill" : "square")
                                    .foregroundColor(selectedAIModels.contains(id) ? .black : .gray)
                
                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                                        Text(name)
                                            .font(.system(size: 14, weight: .medium))
                            .foregroundColor(.black)
                        
                                        if id == "llava" {
                                            Text("較慢")
                                                .font(.system(size: 10))
                                                .foregroundColor(.orange)
                                                .padding(.horizontal, 6)
                                                .padding(.vertical, 2)
                                                .background(
                                                    RoundedRectangle(cornerRadius: 8)
                                                        .fill(Color.orange.opacity(0.2))
                                                )
                }
                
                Spacer()
                                    }
                                    
                                    Text(description)
                                        .font(.system(size: 12))
                        .foregroundColor(.gray)
                }
                                
                                Spacer()
            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(
                                RoundedRectangle(cornerRadius: 10)
                                    .fill(selectedAIModels.contains(id) ? Color.black.opacity(0.05) : Color.clear)
                .overlay(
                                        RoundedRectangle(cornerRadius: 10)
                                            .stroke(selectedAIModels.contains(id) ? Color.black.opacity(0.2) : Color.gray.opacity(0.2), lineWidth: 1)
                                    )
                            )
                        }
                        .buttonStyle(PlainButtonStyle())
                    }
                }
                
                if selectedAIModels.isEmpty {
                    Text("⚠️ 請至少選擇一個AI模型")
                        .font(.system(size: 12))
                        .foregroundColor(.red)
                }
                
                if selectedAIModels.contains("llava") {
                    HStack {
                        Image(systemName: "info.circle")
                            .foregroundColor(.blue)
                        Text("LLaVA模型首次載入需要較長時間，請耐心等待")
                            .font(.system(size: 11))
                            .foregroundColor(.blue)
                    }
                    .padding(.top, 5)
                }
            }
        }
        .padding(.horizontal, 10)
                }
                
    private var analyzeButton: some View {
                Button(action: {
            Task {
                await analyzeOutfit()
                    }
                }) {
            HStack {
                Image(systemName: "sparkles")
                    .font(.system(size: 18, weight: .medium))
                
                Text("開始分析穿搭")
                    .font(.system(size: 18, weight: .bold))
            }
            .foregroundColor(.white)
            .padding(.vertical, 15)
            .padding(.horizontal, 30)
            .background(
                RoundedRectangle(cornerRadius: 25)
                    .fill(Color.black)
            )
            .shadow(color: Color.gray.opacity(0.3), radius: 5, x: 0, y: 2)
        }
    }
    
    private var loadingView: some View {
        VStack(spacing: 15) {
            ProgressView()
                .scaleEffect(1.2)
            
            Text("AI正在分析您的穿搭...")
                .font(.system(size: 16, weight: .medium))
                .foregroundColor(.gray)
            
            Text("這可能需要幾秒鐘時間")
                .font(.system(size: 14))
                .foregroundColor(.gray.opacity(0.8))
                }
        .padding(.vertical, 30)
    }
    
    private var errorView: some View {
        VStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 24))
                .foregroundColor(.red)
            
            Text(errorMessage)
                .font(.system(size: 14, weight: .medium))
                .foregroundColor(.red)
                .multilineTextAlignment(.center)
            
            Button("重試") {
                errorMessage = ""
            }
            .font(.system(size: 14, weight: .medium))
            .foregroundColor(.blue)
            .padding(.top, 5)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color.red.opacity(0.1))
        )
    }
    
    private var recommendationsSection: some View {
        VStack(alignment: .leading, spacing: 20) {
            // 分析結果標題
            HStack {
                Text("分析結果")
                    .font(.system(size: 22, weight: .bold))
                .foregroundColor(.black)
                Spacer()
            }
            
            // 風格分析
            if let analysis = styleAnalysis {
                styleAnalysisCard(analysis)
            }
            
            // 推薦列表
            Text("相似穿搭推薦")
                .font(.system(size: 18, weight: .bold))
                    .foregroundColor(.black)
            
            ForEach(recommendations) { recommendation in
                RecommendationCard(
                    recommendation: recommendation,
                    apiService: apiService,
                    onTap: {
                        selectedRecommendation = recommendation
                        Task {
                            await getAdviceForRecommendation(recommendation)
                        }
                    },
                    isGeneratingAdvice: generatingAdviceForRecommendation.contains(recommendation.recommendationId),
                    hasAdvice: completedAdviceRecommendations.contains(recommendation.recommendationId)
                )
            }
        }
    }
    
    private func styleAnalysisCard(_ analysis: StyleAnalysis) -> some View {
                VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "chart.bar.fill")
                    .foregroundColor(.blue)
                Text("風格分析")
                    .font(.system(size: 16, weight: .bold))
                                .foregroundColor(.black)
            }
            
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("主要風格:")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.gray)
                    Text(styleDisplayName(analysis.dominantStyle))
                        .font(.system(size: 14, weight: .bold))
                                .foregroundColor(.black)
                    }
                    
                    HStack {
                    Text("平均相似度:")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.gray)
                    Text(String(format: "%.1f%%", analysis.averageVisualSimilarity * 100))
                        .font(.system(size: 14, weight: .bold))
                        .foregroundColor(.green)
                }
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.blue.opacity(0.05))
                            .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.blue.opacity(0.2), lineWidth: 1)
                            )
        )
    }
    
    // MARK: - 輔助方法
    private func styleDisplayName(_ style: String) -> String {
        switch style {
        case "": return "不指定"
        case "CASUAL": return "休閒風"
        case "STREET": return "街頭風"
        case "FORMAL": return "正式風"
        case "BOHEMIAN": return "波西米亞風"
        default: return style
        }
    }
    
    private func analyzeOutfit() async {
        guard let image = selectedImage else { return }
        
        isAnalyzing = true
        errorMessage = ""
        
        // 清理之前的建議狀態
        DispatchQueue.main.async {
            self.generatingAdviceForRecommendation.removeAll()
            self.completedAdviceRecommendations.removeAll()
            self.adviceData = nil
        }
        
        do {
            let response = try await apiService.getRecommendations(
                image: image,
                gender: selectedGender,
                stylePreference: selectedStyle.isEmpty ? nil : selectedStyle,
                strategy: selectedStrategy
            )
            
            DispatchQueue.main.async {
                self.recommendations = response.recommendations
                self.styleAnalysis = response.styleAnalysis
                self.userImagePath = response.inputImageUrl
                self.isAnalyzing = false
            }
        } catch {
            DispatchQueue.main.async {
                self.errorMessage = "分析失敗，請檢查網路連接或稍後再試"
                self.isAnalyzing = false
                    }
                }
    }
    
    private func getAdviceForRecommendation(_ recommendation: FashionRecommendation) async {
        let recommendationId = recommendation.recommendationId
        
        // 檢查是否已經在生成建議或已完成
        if generatingAdviceForRecommendation.contains(recommendationId) {
            print("⚠️ 該推薦項目正在生成建議中，跳過重複請求")
            return
        }
        
        if completedAdviceRecommendations.contains(recommendationId) {
            // 如果已經有建議，直接顯示
            print("✅ 建議已存在，直接顯示")
            DispatchQueue.main.async {
                self.showingAdvice = true
            }
            return
        }
        
        // 檢查是否選擇了AI模型
        guard !selectedAIModels.isEmpty else {
            DispatchQueue.main.async {
                self.errorMessage = "請至少選擇一個AI模型"
            }
            return
        }
        
        // 開始生成建議
        DispatchQueue.main.async {
            self.generatingAdviceForRecommendation.insert(recommendationId)
            self.errorMessage = ""
        }
        
        do {
            print("🤖 開始為推薦 \(recommendationId) 生成AI建議...")
            print("📋 使用模型: \(Array(selectedAIModels))")
            print("🎯 目標風格: \(recommendation.style)")
            
            let advice = try await apiService.getAdvice(
                userImagePath: userImagePath,
                targetImagePath: recommendation.path,
                targetStyle: recommendation.style,
                recommendationId: recommendation.recommendationId,
                aiModels: Array(selectedAIModels)
            )
            
            DispatchQueue.main.async {
                // 生成成功，更新狀態
                self.generatingAdviceForRecommendation.remove(recommendationId)
                self.completedAdviceRecommendations.insert(recommendationId)
                self.adviceData = advice
                self.showingAdvice = true
                print("✅ 推薦 \(recommendationId) 的AI建議生成完成")
            }
            
        } catch {
            DispatchQueue.main.async {
                // 生成失敗，重置狀態
                self.generatingAdviceForRecommendation.remove(recommendationId)
                
                // 根據錯誤類型提供更具體的錯誤信息
                if self.selectedAIModels.contains("llava") {
                    self.errorMessage = "AI建議生成失敗。LLaVA模型需要較長載入時間，請稍後再試或使用其他模型。"
                } else {
                    self.errorMessage = "AI建議生成失敗，請檢查網路連接或稍後再試。"
                }
                
                print("❌ 推薦 \(recommendationId) 的AI建議生成失敗: \(error)")
            }
        }
    }
}

// MARK: - 推薦卡片視圖
struct RecommendationCard: View {
    let recommendation: FashionRecommendation
    let apiService: FashionAPIService
    let onTap: () -> Void
    let isGeneratingAdvice: Bool
    let hasAdvice: Bool
    
    var body: some View {
        Button(action: onTap) {
            VStack(spacing: 0) {
                // 推薦圖片 - 更大的展示區域（使用快取）
                CachedAsyncImage(url: apiService.getImageURL(path: recommendation.path)) { phase in
                    switch phase {
                    case .success(let image):
                        image
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                            .frame(width: UIScreen.main.bounds.width - 40, height: 200)
                            .clipped()
                            .cornerRadius(12, corners: [.topLeft, .topRight])
                    case .failure(let error):
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color.red.opacity(0.3))
                            .frame(width: UIScreen.main.bounds.width - 40, height: 200)
                            .overlay(
                                VStack {
                                    Image(systemName: "exclamationmark.triangle")
                                        .font(.system(size: 30))
                                        .foregroundColor(.red)
                                    Text("載入失敗")
                                        .font(.system(size: 14))
                                        .foregroundColor(.red)
                                }
                            )
                            .onAppear {
                                print("❌ 圖片載入失敗: \(recommendation.path), 錯誤: \(error)")
                            }
                    case .empty:
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color.gray.opacity(0.2))
                            .frame(width: UIScreen.main.bounds.width - 40, height: 200)
                            .overlay(
                                VStack {
                                    ProgressView()
                                        .scaleEffect(1.2)
                                    Text("載入中...")
                                        .font(.system(size: 14))
                                        .foregroundColor(.gray)
                                        .padding(.top, 8)
                                }
                            )
                            .onAppear {
                                print("🔄 開始載入圖片: \(recommendation.path)")
                            }
                    @unknown default:
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color.gray.opacity(0.2))
                            .frame(width: UIScreen.main.bounds.width - 40, height: 200)
                    }
                }
                
                // 詳細信息區域
                VStack(alignment: .leading, spacing: 12) {
                    // 風格標題
                    HStack {
                        Text(styleDisplayName(recommendation.style))
                            .font(.system(size: 18, weight: .bold))
                            .foregroundColor(.black)
                        
                        Spacer()
                        
                        // 相似度指示器
                        HStack(spacing: 6) {
                            Circle()
                                .fill(similarityColor(recommendation.similarity))
                                .frame(width: 10, height: 10)
                            Text("\(String(format: "%.1f%%", recommendation.similarity * 100))")
                                .font(.system(size: 14, weight: .bold))
                                .foregroundColor(similarityColor(recommendation.similarity))
                        }
                    }
                    
                    // 數據指標行
                    HStack(spacing: 20) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("視覺相似度")
                                .font(.system(size: 12, weight: .medium))
                                .foregroundColor(.gray)
                            Text("\(String(format: "%.1f%%", recommendation.similarity * 100))")
                                .font(.system(size: 14, weight: .bold))
                                .foregroundColor(.green)
                        }
                        
                        VStack(alignment: .leading, spacing: 4) {
                            Text("推薦評分")
                                .font(.system(size: 12, weight: .medium))
                                .foregroundColor(.gray)
                            Text("\(String(format: "%.1f", recommendation.score))/10")
                                .font(.system(size: 14, weight: .bold))
                                .foregroundColor(.blue)
                        }
                        
                        Spacer()
                    }
                    
                    // AI建議狀態
                    HStack {
                        if isGeneratingAdvice {
                            HStack(spacing: 8) {
                                ProgressView()
                                    .scaleEffect(0.8)
                                Text("AI 正在分析中...")
                                    .font(.system(size: 13, weight: .medium))
                                    .foregroundColor(.orange)
                            }
                        } else if hasAdvice {
                            HStack(spacing: 8) {
                                Image(systemName: "checkmark.circle.fill")
                                    .font(.system(size: 16))
                                    .foregroundColor(.green)
                                Text("點擊查看 AI 建議")
                                    .font(.system(size: 13, weight: .medium))
                                    .foregroundColor(.green)
                            }
                        } else {
                            HStack(spacing: 8) {
                                Image(systemName: "brain.head.profile")
                                    .font(.system(size: 16))
                                    .foregroundColor(.black)
                                Text("點擊獲取 AI 穿搭建議")
                                    .font(.system(size: 13, weight: .medium))
                                    .foregroundColor(.gray)
                            }
                        }
                        
                        Spacer()
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 16)
                .background(Color.white)
            }
            .background(
                RoundedRectangle(cornerRadius: 15)
                    .fill(Color.white)
                    .shadow(color: Color.gray.opacity(0.15), radius: 8, x: 0, y: 4)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 15)
                    .stroke(
                        isGeneratingAdvice ? Color.orange.opacity(0.6) : 
                        hasAdvice ? Color.green.opacity(0.4) : 
                        Color.gray.opacity(0.2), 
                        lineWidth: isGeneratingAdvice || hasAdvice ? 2 : 1
                    )
            )
        }
        .buttonStyle(PlainButtonStyle())
        .disabled(isGeneratingAdvice)
    }
    
    private func styleDisplayName(_ style: String) -> String {
        switch style {
        case "CASUAL": return "休閒風"
        case "STREET": return "街頭風"
        case "FORMAL": return "正式風"
        case "BOHEMIAN": return "波西米亞風"
        default: return style
        }
    }
    
    private func similarityColor(_ similarity: Double) -> Color {
        if similarity >= 0.8 {
            return .green
        } else if similarity >= 0.6 {
            return .orange
        } else {
            return .red
        }
    }
}

// MARK: - 建議詳情視圖
struct AdviceDetailView: View {
    let advice: AdviceResponse
    let recommendation: FashionRecommendation
    @Environment(\.presentationMode) var presentationMode
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // 推薦信息
                    VStack(alignment: .leading, spacing: 10) {
                        Text("推薦風格")
                            .font(.system(size: 20, weight: .bold))
                            .foregroundColor(.black)
                        
                        Text(styleDisplayName(advice.targetStyle))
                            .font(.system(size: 18, weight: .medium))
                            .foregroundColor(.blue)
                            .padding(.horizontal, 15)
                            .padding(.vertical, 8)
                            .background(
                                RoundedRectangle(cornerRadius: 20)
                                    .fill(Color.blue.opacity(0.1))
                            )
                    }
                    
                    // 相似度詳情
                    VStack(alignment: .leading, spacing: 10) {
                        Text("相似度分析")
                            .font(.system(size: 18, weight: .bold))
                            .foregroundColor(.black)
                        
                        VStack(spacing: 8) {
                            HStack {
                                Text("整體相似度:")
                                    .font(.system(size: 14, weight: .medium))
                                    .foregroundColor(.gray)
                                Spacer()
                                Text("\(String(format: "%.1f%%", recommendation.similarity * 100))")
                                    .font(.system(size: 14, weight: .bold))
                                    .foregroundColor(.green)
                            }
                            
                            HStack {
                                Text("視覺相似度:")
                                    .font(.system(size: 14, weight: .medium))
                                    .foregroundColor(.gray)
                                Spacer()
                                Text("\(String(format: "%.1f%%", recommendation.detailedSimilarity.visualSimilarity * 100))")
                                    .font(.system(size: 14, weight: .bold))
                                    .foregroundColor(.blue)
                            }
                            
                            HStack {
                                Text("主要組件相似度:")
                                    .font(.system(size: 14, weight: .medium))
                                    .foregroundColor(.gray)
                                Spacer()
                                Text("\(String(format: "%.1f%%", recommendation.detailedSimilarity.mainComponentSimilarity * 100))")
                                    .font(.system(size: 14, weight: .bold))
                                    .foregroundColor(.purple)
                                }
                            }
                        .padding()
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color.gray.opacity(0.05))
                        )
                    }
                    
                    // AI建議
                    VStack(alignment: .leading, spacing: 15) {
                        Text("AI穿搭建議")
                            .font(.system(size: 20, weight: .bold))
                            .foregroundColor(.black)
                        
                        ForEach(Array(advice.aiAdvice.keys.sorted()), id: \.self) { modelKey in
                            if let adviceText = advice.aiAdvice[modelKey] {
                                AdviceCard(
                                    title: modelDisplayName(modelKey),
                                    content: adviceText,
                                    modelKey: modelKey
                                )
                            }
                        }
                    }
                    
                    Spacer(minLength: 30)
                }
                .padding()
            }
            .navigationTitle("穿搭建議")
            .navigationBarItems(trailing: Button("關閉") {
                presentationMode.wrappedValue.dismiss()
            })
        }
    }
    
    private func styleDisplayName(_ style: String) -> String {
        switch style {
        case "CASUAL": return "休閒風"
        case "STREET": return "街頭風"  
        case "FORMAL": return "正式風"
        case "BOHEMIAN": return "波西米亞風"
        default: return style
        }
    }
    
    private func modelDisplayName(_ model: String) -> String {
        switch model {
        case "rule_based": return "規則系統建議"
        case "clip": return "FashionCLIP分析"
        case "llava": return "視覺語言模型分析"
        default: return model
        }
    }
}

// MARK: - 建議卡片
struct AdviceCard: View {
    let title: String
    let content: String
    let modelKey: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: getModelIcon(modelKey))
                    .foregroundColor(getModelColor(modelKey))
                Text(title)
                    .font(.system(size: 16, weight: .bold))
                    .foregroundColor(.black)
                
                Spacer()
                
                if modelKey == "llava" {
                    Text("深度分析")
                        .font(.system(size: 10))
                        .foregroundColor(.purple)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.purple.opacity(0.1))
                        )
                        }
            }
            
            if modelKey == "rule_based" || content.contains("\n") {
                // 格式化顯示，保留換行
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(content.components(separatedBy: "\n"), id: \.self) { line in
                        if !line.trimmingCharacters(in: .whitespaces).isEmpty {
                            Text(line)
                                .font(.system(size: 14, weight: .regular))
                                .foregroundColor(.gray)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
            } else {
                // 一般文字顯示
                Text(content)
                    .font(.system(size: 14, weight: .regular))
                    .foregroundColor(.gray)
                    .lineSpacing(3)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(getModelBackgroundColor(modelKey))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(getModelColor(modelKey).opacity(0.3), lineWidth: 1)
                )
        )
                        }
                        
    private func getModelIcon(_ model: String) -> String {
        switch model {
        case "rule_based": return "list.bullet"
        case "clip": return "eye.fill"
        case "llava": return "brain.head.profile"
        default: return "sparkles"
        }
    }
    
    private func getModelColor(_ model: String) -> Color {
        switch model {
        case "rule_based": return .blue
        case "clip": return .green
        case "llava": return .purple
        default: return .gray
        }
    }
    
    private func getModelBackgroundColor(_ model: String) -> Color {
        switch model {
        case "rule_based": return .blue.opacity(0.05)
        case "clip": return .green.opacity(0.05)
        case "llava": return .purple.opacity(0.05)
        default: return .gray.opacity(0.05)
                }
            }
        }

// MARK: - 圖片選擇器
struct ImagePickerView: UIViewControllerRepresentable {
    @Binding var selectedImage: UIImage?
    let sourceType: UIImagePickerController.SourceType
    @Environment(\.presentationMode) private var presentationMode
    
    func makeUIViewController(context: UIViewControllerRepresentableContext<ImagePickerView>) -> UIImagePickerController {
        let imagePicker = UIImagePickerController()
        imagePicker.allowsEditing = false
        imagePicker.sourceType = sourceType
        imagePicker.delegate = context.coordinator
        
        // 確保有正確的背景
        imagePicker.view.backgroundColor = UIColor.systemBackground
        imagePicker.modalPresentationStyle = .fullScreen
        
        // 檢查相機是否可用（針對拍照功能）
        if sourceType == .camera && !UIImagePickerController.isSourceTypeAvailable(.camera) {
            // 如果相機不可用，回退到照片庫
            imagePicker.sourceType = .photoLibrary
        }
        
        return imagePicker
    }
    
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: UIViewControllerRepresentableContext<ImagePickerView>) {
        // 確保每次更新時都有正確的背景
        uiViewController.view.backgroundColor = UIColor.systemBackground
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        var parent: ImagePickerView
        
        init(_ parent: ImagePickerView) {
            self.parent = parent
        }
        
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            if let image = info[UIImagePickerController.InfoKey.originalImage] as? UIImage {
                parent.selectedImage = image
            }
            parent.presentationMode.wrappedValue.dismiss()
        }
        
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.presentationMode.wrappedValue.dismiss()
        }
    }
}

struct FeedView_Previews: PreviewProvider {
    static var previews: some View {
    FeedView()
    }
} 