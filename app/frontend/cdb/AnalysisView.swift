//
//  AnalysisView.swift
//  cdb
//
//  Created by kevin Chou on 2025/4/28.
//

import SwiftUI

struct AnalysisView: View {
    let selectedImage: UIImage
    let onDismiss: () -> Void
    
    @StateObject private var apiService = FashionAPIService()
    @State private var selectedGender = "MEN"
    @State private var selectedStyle = ""
    @State private var selectedStrategy = "balanced"
    @State private var isAnalyzing = false
    @State private var recommendations: [FashionRecommendation] = []
    @State private var styleAnalysis: StyleAnalysis?
    @State private var errorMessage = ""
    @State private var currentRecommendationIndex = 0
    @State private var userImagePath = ""
    @State private var selectedAIModels: Set<String> = ["rule_based", "clip"]
    @State private var generatingAdviceForRecommendation: Set<String> = []
    @State private var completedAdviceRecommendations: Set<String> = []
    @State private var showingAdvice = false
    @State private var adviceData: AdviceResponse?
    @State private var selectedRecommendation: FashionRecommendation?
    
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
        GeometryReader { geometry in
            ZStack {
                // 全黑背景
                Color.black
                    .ignoresSafeArea()
                
                if isAnalyzing {
                    // 分析中的視圖
                    analysisLoadingView
                } else if !recommendations.isEmpty {
                    // 推薦結果視圖
                    recommendationCardsView(geometry: geometry)
                } else {
                    // 設置視圖
                    analysisSettingsView(geometry: geometry)
                }
                
                // 頂部導航欄
                VStack {
                    topNavigationBar
                    Spacer()
                }
                
                // 錯誤提示
                if !errorMessage.isEmpty {
                    VStack {
                        Spacer()
                        errorBanner
                        Spacer().frame(height: 100)
                    }
                }
            }
        }
        .navigationBarHidden(true)
        .sheet(isPresented: $showingAdvice) {
            if let advice = adviceData, let recommendation = selectedRecommendation {
                ModernAdviceView(advice: advice, recommendation: recommendation)
            }
        }
    }
    
    // MARK: - 頂部導航欄
    private var topNavigationBar: some View {
        HStack {
            Button(action: onDismiss) {
                Image(systemName: "chevron.left")
                    .font(.system(size: 18, weight: .medium))
                    .foregroundColor(.white)
                    .frame(width: 44, height: 44)
                    .background(Color.black.opacity(0.3))
                    .clipShape(Circle())
            }
            
            Spacer()
            
            if !recommendations.isEmpty {
                Text("推薦結果")
                    .font(.system(size: 18, weight: .bold))
                    .foregroundColor(.white)
            } else if isAnalyzing {
                Text("分析中...")
                    .font(.system(size: 18, weight: .bold))
                    .foregroundColor(.white)
            } else {
                Text("設置分析")
                    .font(.system(size: 18, weight: .bold))
                    .foregroundColor(.white)
            }
            
            Spacer()
            
            if !recommendations.isEmpty {
                Text("\(currentRecommendationIndex + 1)/\(recommendations.count)")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(.white.opacity(0.8))
                    .frame(width: 44, height: 44)
            } else {
                Spacer().frame(width: 44, height: 44)
            }
        }
        .padding(.horizontal, 20)
        .padding(.top, 10)
    }
    
    // MARK: - 分析設置視圖
    private func analysisSettingsView(geometry: GeometryProxy) -> some View {
        ScrollView {
            VStack(spacing: 30) {
                Spacer().frame(height: 80)
                
                // 用戶圖片預覽
                userImagePreview(geometry: geometry)
                
                // 設置選項
                settingsPanel
                
                // 分析按鈕
                analyzeButton
                
                Spacer().frame(height: 50)
            }
        }
    }
    
    // MARK: - 用戶圖片預覽
    private func userImagePreview(geometry: GeometryProxy) -> some View {
        VStack(spacing: 15) {
            Text("您的穿搭")
                .font(.system(size: 20, weight: .bold))
                .foregroundColor(.white)
            
            Image(uiImage: selectedImage)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(maxWidth: min(geometry.size.width * 0.7, 300))
                .frame(maxHeight: min(geometry.size.height * 0.4, 400))
                .cornerRadius(20)
                .shadow(color: .black.opacity(0.3), radius: 10, x: 0, y: 5)
        }
    }
    
    // MARK: - 設置面板
    private var settingsPanel: some View {
        VStack(spacing: 25) {
            // 性別選擇
            SettingCard(title: "性別", icon: "person.fill") {
                HStack {
                    ForEach(genders, id: \.self) { gender in
                        Button(action: {
                            selectedGender = gender
                        }) {
                            Text(gender == "MEN" ? "男性" : "女性")
                                .font(.system(size: 14, weight: .medium))
                                .foregroundColor(selectedGender == gender ? .black : .white)
                                .padding(.horizontal, 20)
                                .padding(.vertical, 10)
                                .background(
                                    RoundedRectangle(cornerRadius: 20)
                                        .fill(selectedGender == gender ? Color.white : Color.white.opacity(0.2))
                                )
                        }
                    }
                    Spacer()
                }
            }
            
            // 風格選擇
            SettingCard(title: "風格偏好", subtitle: "可選", icon: "paintbrush.fill") {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack {
                        ForEach(styles, id: \.self) { style in
                            Button(action: {
                                selectedStyle = style
                            }) {
                                Text(styleDisplayName(style))
                                    .font(.system(size: 14, weight: .medium))
                                    .foregroundColor(selectedStyle == style ? .black : .white)
                                    .padding(.horizontal, 16)
                                    .padding(.vertical, 8)
                                    .background(
                                        RoundedRectangle(cornerRadius: 16)
                                            .fill(selectedStyle == style ? Color.white : Color.white.opacity(0.2))
                                    )
                            }
                        }
                    }
                    .padding(.horizontal, 5)
                }
            }
            
            // AI模型選擇
            SettingCard(title: "AI分析模型", icon: "brain.head.profile") {
                VStack(spacing: 12) {
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
                                Image(systemName: selectedAIModels.contains(id) ? "checkmark.circle.fill" : "circle")
                                    .foregroundColor(selectedAIModels.contains(id) ? .white : .white.opacity(0.6))
                                
                                VStack(alignment: .leading, spacing: 4) {
                                    HStack {
                                        Text(name)
                                            .font(.system(size: 14, weight: .medium))
                                            .foregroundColor(.white)
                                        
                                        if id == "llava" {
                                            Text("較慢")
                                                .font(.system(size: 10))
                                                .foregroundColor(.orange)
                                                .padding(.horizontal, 6)
                                                .padding(.vertical, 2)
                                                .background(
                                                    RoundedRectangle(cornerRadius: 8)
                                                        .fill(Color.orange.opacity(0.3))
                                                )
                                        }
                                        
                                        Spacer()
                                    }
                                    
                                    Text(description)
                                        .font(.system(size: 12))
                                        .foregroundColor(.white.opacity(0.7))
                                }
                                
                                Spacer()
                            }
                            .padding(.horizontal, 15)
                            .padding(.vertical, 12)
                            .background(
                                RoundedRectangle(cornerRadius: 12)
                                    .fill(selectedAIModels.contains(id) ? Color.white.opacity(0.15) : Color.clear)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 12)
                                            .stroke(Color.white.opacity(0.3), lineWidth: 1)
                                    )
                            )
                        }
                        .buttonStyle(PlainButtonStyle())
                    }
                    
                    if selectedAIModels.contains("llava") {
                        HStack {
                            Image(systemName: "info.circle")
                                .foregroundColor(.blue)
                            Text("LLaVA模型首次載入需要較長時間")
                                .font(.system(size: 11))
                                .foregroundColor(.blue)
                        }
                        .padding(.top, 5)
                    }
                }
            }
        }
        .padding(.horizontal, 20)
    }
    
    // MARK: - 分析按鈕
    private var analyzeButton: some View {
        Button(action: {
            Task {
                await analyzeOutfit()
            }
        }) {
            HStack {
                Image(systemName: "sparkles")
                    .font(.system(size: 18, weight: .medium))
                
                Text("開始AI分析")
                    .font(.system(size: 18, weight: .bold))
            }
            .foregroundColor(.black)
            .padding(.vertical, 16)
            .padding(.horizontal, 40)
            .background(
                RoundedRectangle(cornerRadius: 25)
                    .fill(Color.white)
            )
            .shadow(color: Color.black.opacity(0.3), radius: 10, x: 0, y: 5)
        }
        .disabled(selectedAIModels.isEmpty)
        .opacity(selectedAIModels.isEmpty ? 0.6 : 1.0)
    }
    
    // MARK: - 分析加載視圖
    private var analysisLoadingView: some View {
        VStack(spacing: 30) {
            Spacer()
            
            // 加載動畫
            VStack(spacing: 20) {
                ZStack {
                    Circle()
                        .stroke(Color.white.opacity(0.3), lineWidth: 4)
                        .frame(width: 80, height: 80)
                    
                    Circle()
                        .trim(from: 0, to: 0.7)
                        .stroke(Color.white, lineWidth: 4)
                        .frame(width: 80, height: 80)
                        .rotationEffect(.degrees(-90))
                        .rotationEffect(.degrees(isAnalyzing ? 360 : 0))
                        .animation(.linear(duration: 1).repeatForever(autoreverses: false), value: isAnalyzing)
                }
                
                Text("AI正在分析您的穿搭...")
                    .font(.system(size: 18, weight: .medium))
                    .foregroundColor(.white)
                
                Text("請稍候，這可能需要幾秒鐘")
                    .font(.system(size: 14))
                    .foregroundColor(.white.opacity(0.7))
            }
            
            Spacer()
        }
    }
    
    // MARK: - 推薦卡片視圖
    private func recommendationCardsView(geometry: GeometryProxy) -> some View {
        VStack(spacing: 0) {
            Spacer().frame(height: 70)
            
            // 卡片滑動視圖
            TabView(selection: $currentRecommendationIndex) {
                ForEach(recommendations.indices, id: \.self) { index in
                    ScrollView {
                        RecommendationCard3D(
                            recommendation: recommendations[index],
                            userImage: selectedImage,
                            apiService: apiService,
                            geometry: geometry,
                            onGenerateAdvice: {
                                selectedRecommendation = recommendations[index]
                                Task {
                                    await getAdviceForRecommendation(recommendations[index])
                                }
                            },
                            isGeneratingAdvice: generatingAdviceForRecommendation.contains(recommendations[index].recommendationId),
                            hasAdvice: completedAdviceRecommendations.contains(recommendations[index].recommendationId)
                        )
                    }
                    .tag(index)
                }
            }
            .tabViewStyle(PageTabViewStyle(indexDisplayMode: .never))
            .frame(height: min(geometry.size.height * 0.85, 800))
            .clipped()
            
            // 卡片指示器
            HStack(spacing: 8) {
                ForEach(recommendations.indices, id: \.self) { index in
                    Circle()
                        .fill(currentRecommendationIndex == index ? Color.white : Color.white.opacity(0.4))
                        .frame(width: 8, height: 8)
                        .scaleEffect(currentRecommendationIndex == index ? 1.2 : 1.0)
                        .animation(.easeInOut(duration: 0.3), value: currentRecommendationIndex)
                }
            }
            .padding(.top, 10)
            .padding(.bottom, 20)
            
            Spacer()
        }
    }
    
    // MARK: - 錯誤橫幅
    private var errorBanner: some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.red)
            
            Text(errorMessage)
                .font(.system(size: 14, weight: .medium))
                .foregroundColor(.white)
                .multilineTextAlignment(.leading)
            
            Spacer()
            
            Button("重試") {
                errorMessage = ""
            }
            .font(.system(size: 14, weight: .medium))
            .foregroundColor(.blue)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.red.opacity(0.2))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.red.opacity(0.5), lineWidth: 1)
                )
        )
        .padding(.horizontal, 20)
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
                image: selectedImage,
                gender: selectedGender,
                stylePreference: selectedStyle.isEmpty ? nil : selectedStyle,
                strategy: selectedStrategy
            )
            
            DispatchQueue.main.async {
                self.recommendations = response.recommendations
                self.styleAnalysis = response.styleAnalysis
                self.userImagePath = response.inputImageUrl
                self.isAnalyzing = false
                self.currentRecommendationIndex = 0
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
            return
        }
        
        if completedAdviceRecommendations.contains(recommendationId) {
            DispatchQueue.main.async {
                self.showingAdvice = true
            }
            return
        }
        
        guard !selectedAIModels.isEmpty else {
            DispatchQueue.main.async {
                self.errorMessage = "請至少選擇一個AI模型"
            }
            return
        }
        
        DispatchQueue.main.async {
            self.generatingAdviceForRecommendation.insert(recommendationId)
            self.errorMessage = ""
        }
        
        do {
            let advice = try await apiService.getAdvice(
                userImagePath: userImagePath,
                targetImagePath: recommendation.path,
                targetStyle: recommendation.style,
                recommendationId: recommendation.recommendationId,
                aiModels: Array(selectedAIModels)
            )
            
            DispatchQueue.main.async {
                self.generatingAdviceForRecommendation.remove(recommendationId)
                self.completedAdviceRecommendations.insert(recommendationId)
                self.adviceData = advice
                self.showingAdvice = true
            }
            
        } catch {
            DispatchQueue.main.async {
                self.generatingAdviceForRecommendation.remove(recommendationId)
                
                // 詳細的錯誤處理
                if let apiError = error as? APIError {
                    switch apiError {
                    case .llavaTimeout:
                        self.errorMessage = "LLaVA模型響應超時，請稍後再試。建議：可先嘗試其他AI模型。"
                    case .decodingError:
                        self.errorMessage = "AI建議已生成，但解析時出現問題。請重新嘗試或查看詳細日誌。"
                    case .serverError:
                        self.errorMessage = "服務器連接錯誤，請檢查網路連接或稍後再試。"
                    case .invalidURL:
                        self.errorMessage = "API配置錯誤，請聯繫技術支持。"
                    }
                } else {
                    if self.selectedAIModels.contains("llava") {
                        self.errorMessage = "AI建議生成失敗。LLaVA模型需要較長載入時間，請稍後再試或使用其他模型。"
                    } else {
                        self.errorMessage = "AI建議生成失敗，請檢查網路連接或稍後再試。"
                    }
                }
                
                print("❌ AI建議生成錯誤: \(error)")
            }
        }
    }
}

// MARK: - 設置卡片組件
struct SettingCard<Content: View>: View {
    let title: String
    let subtitle: String?
    let icon: String
    let content: Content
    
    init(title: String, subtitle: String? = nil, icon: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.subtitle = subtitle
        self.icon = icon
        self.content = content()
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(.white)
                    .frame(width: 20)
                
                Text(title)
                    .font(.system(size: 16, weight: .bold))
                    .foregroundColor(.white)
                
                if let subtitle = subtitle {
                    Text("(\(subtitle))")
                        .font(.system(size: 14))
                        .foregroundColor(.white.opacity(0.6))
                }
                
                Spacer()
            }
            
            content
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color.white.opacity(0.1))
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(Color.white.opacity(0.2), lineWidth: 1)
                )
        )
    }
} 