//
//  ModernAdviceView.swift
//  cdb
//
//  Created by kevin Chou on 2025/4/28.
//

import SwiftUI

struct ModernAdviceView: View {
    let advice: AdviceResponse
    let recommendation: FashionRecommendation
    @Environment(\.presentationMode) var presentationMode
    @State private var selectedModelIndex = 0
    
    private var availableModels: [(String, String)] {
        advice.aiAdvice.map { (key, _) in
            let displayName = modelDisplayName(key)
            return (key, displayName)
        }.sorted { $0.1 < $1.1 }
    }
    
    var body: some View {
        NavigationView {
            GeometryReader { geometry in
                ZStack {
                    // 純黑背景
                    Color.black
                        .ignoresSafeArea()
                    
                    ScrollView {
                        VStack(spacing: 25) {
                            // 頂部標題區域
                            headerSection
                            
                            // 風格信息卡片
                            styleInfoCard
                            
                            // 相似度分析
                            similarityAnalysisCard
                            
                            // AI模型選擇器
                            if availableModels.count > 1 {
                                modelSelector
                            }
                            
                            // AI建議內容
                            adviceContentCard
                            
                            Spacer(minLength: 50)
                        }
                        .padding(.horizontal, 20)
                        .padding(.top, 20)
                    }
                }
            }
            .navigationBarHidden(true)
        }
    }
    
    // MARK: - 頂部標題區域
    private var headerSection: some View {
        VStack(spacing: 15) {
            HStack {
                Button(action: {
                    presentationMode.wrappedValue.dismiss()
                }) {
                    Image(systemName: "chevron.down")
                        .font(.system(size: 18, weight: .medium))
                        .foregroundColor(.white)
                        .frame(width: 44, height: 44)
                        .background(Color.white.opacity(0.2))
                        .clipShape(Circle())
                }
                
                Spacer()
                
                Text("AI穿搭建議")
                    .font(.system(size: 20, weight: .bold))
                    .foregroundColor(.white)
                
                Spacer()
                
                Button(action: {
                    // 分享功能
                }) {
                    Image(systemName: "square.and.arrow.up")
                        .font(.system(size: 18, weight: .medium))
                        .foregroundColor(.white)
                        .frame(width: 44, height: 44)
                        .background(Color.white.opacity(0.2))
                        .clipShape(Circle())
                }
            }
            
            Text("專業AI為您量身打造的穿搭建議")
                .font(.system(size: 14))
                .foregroundColor(.white.opacity(0.8))
                .multilineTextAlignment(.center)
        }
    }
    
    // MARK: - 風格信息卡片
    private var styleInfoCard: some View {
        VStack(spacing: 15) {
            HStack {
                Image(systemName: styleIcon(advice.targetStyle))
                    .font(.system(size: 24, weight: .medium))
                    .foregroundColor(.blue)
                
                VStack(alignment: .leading, spacing: 4) {
                    Text("目標風格")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.white.opacity(0.7))
                    
                    Text(styleDisplayName(advice.targetStyle))
                        .font(.system(size: 20, weight: .bold))
                        .foregroundColor(.white)
                }
                
                Spacer()
                
                VStack(alignment: .trailing, spacing: 4) {
                    Text("分析時間")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundColor(.white.opacity(0.7))
                    
                    Text(String(format: "%.1fs", advice.analysisTime))
                        .font(.system(size: 16, weight: .bold))
                        .foregroundColor(.green)
                }
            }
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 20)
                .fill(Color.white.opacity(0.15))
                .overlay(
                    RoundedRectangle(cornerRadius: 20)
                        .stroke(Color.white.opacity(0.2), lineWidth: 1)
                )
        )
    }
    
    // MARK: - 相似度分析卡片
    private var similarityAnalysisCard: some View {
        VStack(spacing: 20) {
            HStack {
                Image(systemName: "chart.bar.fill")
                    .font(.system(size: 18))
                    .foregroundColor(.purple)
                
                Text("相似度分析")
                    .font(.system(size: 18, weight: .bold))
                    .foregroundColor(.white)
                
                Spacer()
            }
            
            VStack(spacing: 15) {
                similarityRow(
                    title: "整體相似度",
                    value: recommendation.similarity,
                    color: .blue,
                    icon: "target"
                )
                
                similarityRow(
                    title: "視覺相似度",
                    value: recommendation.detailedSimilarity.visualSimilarity,
                    color: .green,
                    icon: "eye.fill"
                )
                
                similarityRow(
                    title: "組件相似度",
                    value: recommendation.detailedSimilarity.mainComponentSimilarity,
                    color: .orange,
                    icon: "puzzlepiece.fill"
                )
                
                if let styleSimilarity = recommendation.detailedSimilarity.styleSimilarity {
                    similarityRow(
                        title: "風格相似度",
                        value: styleSimilarity,
                        color: .purple,
                        icon: "paintbrush.fill"
                    )
                }
            }
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 20)
                .fill(Color.white.opacity(0.1))
                .overlay(
                    RoundedRectangle(cornerRadius: 20)
                        .stroke(Color.white.opacity(0.2), lineWidth: 1)
                )
        )
    }
    
    // MARK: - AI模型選擇器
    private var modelSelector: some View {
        VStack(spacing: 15) {
            HStack {
                Image(systemName: "brain.head.profile")
                    .font(.system(size: 18))
                    .foregroundColor(.cyan)
                
                Text("AI模型")
                    .font(.system(size: 18, weight: .bold))
                    .foregroundColor(.white)
                
                Spacer()
            }
            
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    ForEach(availableModels.indices, id: \.self) { index in
                        let (_, displayName) = availableModels[index]
                        
                        Button(action: {
                            selectedModelIndex = index
                        }) {
                            Text(displayName)
                                .font(.system(size: 14, weight: .medium))
                                .foregroundColor(selectedModelIndex == index ? .black : .white)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 10)
                                .background(
                                    RoundedRectangle(cornerRadius: 20)
                                        .fill(selectedModelIndex == index ? Color.white : Color.white.opacity(0.2))
                                )
                        }
                    }
                }
                .padding(.horizontal, 5)
            }
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 20)
                .fill(Color.white.opacity(0.1))
                .overlay(
                    RoundedRectangle(cornerRadius: 20)
                        .stroke(Color.white.opacity(0.2), lineWidth: 1)
                )
        )
    }
    
    // MARK: - AI建議內容卡片
    private var adviceContentCard: some View {
        VStack(spacing: 20) {
            HStack {
                Image(systemName: modelIcon(getCurrentModelKey()))
                    .font(.system(size: 18))
                    .foregroundColor(modelColor(getCurrentModelKey()))
                
                Text("\(getCurrentModelDisplayName())建議")
                    .font(.system(size: 18, weight: .bold))
                    .foregroundColor(.white)
                
                Spacer()
                
                // 建議評級
                HStack(spacing: 4) {
                    ForEach(0..<5) { index in
                        Image(systemName: index < getAdviceRating() ? "star.fill" : "star")
                            .font(.system(size: 12))
                            .foregroundColor(.yellow)
                    }
                }
            }
            
            // 建議內容
            adviceContent
        }
        .padding(20)
        .background(
            RoundedRectangle(cornerRadius: 20)
                .fill(Color.white.opacity(0.1))
                .overlay(
                    RoundedRectangle(cornerRadius: 20)
                        .stroke(modelColor(getCurrentModelKey()).opacity(0.5), lineWidth: 1)
                )
        )
    }
    
    // MARK: - 建議內容
    private var adviceContent: some View {
        let currentAdvice = getCurrentAdviceText()
        
        return VStack(alignment: .leading, spacing: 12) {
            if currentAdvice.contains("\n") || currentAdvice.contains("•") || currentAdvice.contains("1.") {
                // 格式化顯示，保留結構
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(formatAdviceText(currentAdvice), id: \.self) { line in
                        if !line.trimmingCharacters(in: .whitespaces).isEmpty {
                            HStack(alignment: .top, spacing: 10) {
                                if line.hasPrefix("•") || line.contains(":") {
                                    Circle()
                                        .fill(modelColor(getCurrentModelKey()))
                                        .frame(width: 6, height: 6)
                                        .padding(.top, 8)
                                }
                                
                                Text(line)
                                    .font(.system(size: 15, weight: .regular))
                                    .foregroundColor(.white.opacity(0.9))
                                    .lineSpacing(4)
                                    .fixedSize(horizontal: false, vertical: true)
                                
                                Spacer()
                            }
                        }
                    }
                }
            } else {
                // 一般文字顯示
                Text(currentAdvice)
                    .font(.system(size: 15, weight: .regular))
                    .foregroundColor(.white.opacity(0.9))
                    .lineSpacing(6)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
    
    // MARK: - 輔助組件
    private func similarityRow(title: String, value: Double, color: Color, icon: String) -> some View {
        HStack {
            Image(systemName: icon)
                .font(.system(size: 14))
                .foregroundColor(color)
                .frame(width: 20)
            
            Text(title)
                .font(.system(size: 14, weight: .medium))
                .foregroundColor(.white.opacity(0.8))
            
            Spacer()
            
            Text("\(Int(value * 100))%")
                .font(.system(size: 14, weight: .bold))
                .foregroundColor(color)
            
            // 進度條
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(color.opacity(0.3))
                        .frame(height: 6)
                    
                    RoundedRectangle(cornerRadius: 4)
                        .fill(color)
                        .frame(width: geo.size.width * value, height: 6)
                }
            }
            .frame(width: 60, height: 6)
        }
    }
    
    // MARK: - 輔助方法
    private func getCurrentModelKey() -> String {
        guard selectedModelIndex < availableModels.count else { return availableModels.first?.0 ?? "" }
        return availableModels[selectedModelIndex].0
    }
    
    private func getCurrentModelDisplayName() -> String {
        guard selectedModelIndex < availableModels.count else { return availableModels.first?.1 ?? "" }
        return availableModels[selectedModelIndex].1
    }
    
    private func getCurrentAdviceText() -> String {
        let modelKey = getCurrentModelKey()
        return advice.aiAdvice[modelKey] ?? "暫無建議"
    }
    
    private func formatAdviceText(_ text: String) -> [String] {
        return text.components(separatedBy: "\n")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }
    
    private func getAdviceRating() -> Int {
        // 根據相似度和建議長度評級
        let similarity = recommendation.similarity
        let adviceLength = getCurrentAdviceText().count
        
        if similarity > 0.8 && adviceLength > 100 {
            return 5
        } else if similarity > 0.7 && adviceLength > 80 {
            return 4
        } else if similarity > 0.6 && adviceLength > 60 {
            return 3
        } else if similarity > 0.5 && adviceLength > 40 {
            return 2
        } else {
            return 1
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
    
    private func styleIcon(_ style: String) -> String {
        switch style {
        case "CASUAL": return "tshirt.fill"
        case "STREET": return "skateboard"
        case "FORMAL": return "suit.heart.fill"
        case "BOHEMIAN": return "leaf.fill"
        default: return "tag.fill"
        }
    }
    
    private func modelDisplayName(_ model: String) -> String {
        switch model {
        case "rule_based": return "規則系統"
        case "clip": return "FashionCLIP"
        case "llava": return "視覺語言模型"
        case "blip2": return "圖像描述模型"
        case "instructblip": return "指令模型"
        default: return model
        }
    }
    
    private func modelIcon(_ model: String) -> String {
        switch model {
        case "rule_based": return "list.bullet.circle.fill"
        case "clip": return "eye.circle.fill"
        case "llava": return "brain.head.profile.fill"
        case "blip2": return "photo.circle.fill"
        case "instructblip": return "command.circle.fill"
        default: return "sparkles.circle.fill"
        }
    }
    
    private func modelColor(_ model: String) -> Color {
        switch model {
        case "rule_based": return .blue
        case "clip": return .green
        case "llava": return .purple
        case "blip2": return .orange
        case "instructblip": return .pink
        default: return .gray
        }
    }
}

struct ModernAdviceView_Previews: PreviewProvider {
    static var previews: some View {
        ModernAdviceView(
            advice: AdviceResponse(
                status: "success",
                requestId: "test",
                recommendationId: "test",
                targetStyle: "CASUAL",
                aiAdvice: [
                    "rule_based": "建議調整上衣顏色搭配，選擇更休閒的款式",
                    "clip": "顏色協調性良好，建議加入休閒配飾"
                ],
                analysisTime: 2.5
            ),
            recommendation: FashionRecommendation(
                recommendationId: "test",
                path: "/test/path",
                style: "CASUAL",
                gender: "MEN",
                similarity: 0.85,
                score: 8.5,
                detailedSimilarity: DetailedSimilarity(
                    visualSimilarity: 0.85,
                    mainComponentSimilarity: 0.78,
                    styleSimilarity: 0.92
                )
            )
        )
    }
} 