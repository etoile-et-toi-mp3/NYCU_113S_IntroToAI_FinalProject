//
//  RecommendationCard3D.swift
//  cdb
//
//  Created by kevin Chou on 2025/4/28.
//

import SwiftUI

struct RecommendationCard3D: View {
    let recommendation: FashionRecommendation
    let userImage: UIImage
    let apiService: FashionAPIService
    let geometry: GeometryProxy
    let onGenerateAdvice: () -> Void
    let isGeneratingAdvice: Bool
    let hasAdvice: Bool
    
    @State private var isPressed = false
    
    var body: some View {
        // 主卡片
        mainCard
            .scaleEffect(isPressed ? 0.98 : 1.0)
            .animation(.easeInOut(duration: 0.2), value: isPressed)
            .onTapGesture {
                onGenerateAdvice()
            }
            .onLongPressGesture(minimumDuration: 0.1) {
                isPressed = false
            } onPressingChanged: { isPressing in
                isPressed = isPressing
            }
            .frame(width: min(geometry.size.width * 0.85, 320))
            .frame(minHeight: 550) // 減少最小高度
            .padding(.vertical, 10) // 添加垂直間距
    }
    
    private var mainCard: some View {
        VStack(spacing: 0) {
            // 推薦圖片區域（更大）
            recommendationImageSection
            
            // 信息區域
            informationSection
            
            // 操作按鈕區域
            actionSection
        }
        .background(
            RoundedRectangle(cornerRadius: 25)
                .fill(
                    LinearGradient(
                        colors: [
                            Color.black.opacity(0.9),
                            Color.black.opacity(0.95),
                            Color.black
                        ],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .shadow(color: .white.opacity(0.1), radius: 20, x: 0, y: 10)
                .overlay(
                    RoundedRectangle(cornerRadius: 25)
                        .stroke(
                            LinearGradient(
                                colors: [Color.white.opacity(0.3), Color.clear],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 1
                        )
                )
        )
    }
    
    // MARK: - 推薦圖片區域（新設計）
    private var recommendationImageSection: some View {
        VStack(spacing: 15) {
            // 標題
            Text("推薦穿搭")
                .font(.system(size: 22, weight: .bold))
                .foregroundColor(.white)
                .padding(.top, 25)
            
            // 推薦圖片（適應不同縱橫比，帶快取）
            CachedAsyncImage(url: apiService.getImageURL(path: recommendation.path)) { phase in
                switch phase {
                case .success(let image):
                    image
                        .resizable()
                        .aspectRatio(contentMode: .fit) // 保持完整圖片比例
                        .frame(maxWidth: 280, maxHeight: 350)
                        .background(
                            RoundedRectangle(cornerRadius: 16)
                                .fill(Color.gray.opacity(0.1))
                        )
                        .cornerRadius(16)
                case .failure(_):
                    VStack(spacing: 10) {
                        Image(systemName: "photo")
                            .font(.system(size: 25))
                            .foregroundColor(.gray)
                        Text("載入失敗")
                            .font(.system(size: 12))
                            .foregroundColor(.gray)
                    }
                    .frame(width: 280, height: 200)
                    .background(
                        RoundedRectangle(cornerRadius: 16)
                            .fill(Color.gray.opacity(0.1))
                    )
                case .empty:
                    VStack(spacing: 8) {
                        ProgressView()
                            .scaleEffect(1.0)
                            .tint(.white)
                        Text("載入中...")
                            .font(.system(size: 12))
                            .foregroundColor(.white.opacity(0.7))
                    }
                    .frame(width: 280, height: 200)
                    .background(
                        RoundedRectangle(cornerRadius: 16)
                            .fill(Color.gray.opacity(0.1))
                    )
                @unknown default:
                    Rectangle()
                        .fill(Color.gray.opacity(0.3))
                        .frame(width: 280, height: 200)
                        .cornerRadius(16)
                }
            }
        }
        .padding(.horizontal, 15)
    }
    
    // MARK: - 信息區域
    private var informationSection: some View {
        VStack(spacing: 10) {
            Divider()
                .background(Color.white.opacity(0.3))
                .padding(.horizontal, 15)
            
            // 風格標籤（移除評分）
            HStack {
                Image(systemName: styleIcon(recommendation.style))
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(.blue)
                
                Text(styleDisplayName(recommendation.style))
                    .font(.system(size: 18, weight: .bold))
                    .foregroundColor(.white)
                
                Spacer()
            }
            .padding(.horizontal, 15)
            
            // 詳細相似度信息
            VStack(spacing: 6) {
                similarityBar(title: "視覺相似度", value: recommendation.detailedSimilarity.visualSimilarity, color: .blue)
                similarityBar(title: "組件相似度", value: recommendation.detailedSimilarity.mainComponentSimilarity, color: .green)
                
                if let styleSimilarity = recommendation.detailedSimilarity.styleSimilarity {
                    similarityBar(title: "風格相似度", value: styleSimilarity, color: .purple)
                }
            }
            .padding(.horizontal, 15)
        }
    }
    
    // MARK: - 操作按鈕區域
    private var actionSection: some View {
        VStack(spacing: 12) {
            Divider()
                .background(Color.white.opacity(0.3))
                .padding(.horizontal, 15)
            
            Button(action: onGenerateAdvice) {
                HStack {
                    if isGeneratingAdvice {
                        ProgressView()
                            .scaleEffect(0.8)
                            .foregroundColor(.white)
                    } else if hasAdvice {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 16))
                            .foregroundColor(.white)
                    } else {
                        Image(systemName: "brain.head.profile")
                            .font(.system(size: 16))
                            .foregroundColor(.white)
                    }
                    
                    Text(actionButtonText)
                        .font(.system(size: 14, weight: .bold))
                        .foregroundColor(.white)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
                .background(
                    RoundedRectangle(cornerRadius: 18)
                        .fill(
                            LinearGradient(
                                colors: actionButtonColors,
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                )
                .scaleEffect(isPressed ? 0.98 : 1.0)
            }
            .disabled(isGeneratingAdvice)
            .padding(.horizontal, 15)
            .padding(.bottom, 15)
        }
    }
    
    // MARK: - 輔助組件
    private func similarityBar(title: String, value: Double, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.white.opacity(0.7))
                
                Spacer()
                
                Text("\(Int(value * 100))%")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundColor(color)
            }
            
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
            .frame(height: 6)
        }
    }
    
    // MARK: - 輔助方法
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
    
    private func similarityColor(_ similarity: Double) -> Color {
        if similarity >= 0.8 {
            return .green
        } else if similarity >= 0.6 {
            return .orange
        } else {
            return .red
        }
    }
    
    private var actionButtonText: String {
        if isGeneratingAdvice {
            return "AI分析中..."
        } else if hasAdvice {
            return "查看詳細建議"
        } else {
            return "獲取AI建議"
        }
    }
    
    private var actionButtonColors: [Color] {
        if isGeneratingAdvice {
            return [Color.orange.opacity(0.8), Color.orange]
        } else if hasAdvice {
            return [Color.green.opacity(0.8), Color.green]
        } else {
            return [Color.blue.opacity(0.8), Color.blue]
        }
    }
}

struct RecommendationCard3D_Previews: PreviewProvider {
    static var previews: some View {
        GeometryReader { geometry in
            RecommendationCard3D(
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
                ),
                userImage: UIImage(systemName: "person.fill") ?? UIImage(),
                apiService: FashionAPIService(),
                geometry: geometry,
                onGenerateAdvice: {},
                isGeneratingAdvice: false,
                hasAdvice: false
            )
        }
        .background(Color.black.opacity(0.5))
    }
} 