//
//  SwipeView.swift
//  cdb
//
//  Created by kevin Chou on 2025/4/28.
//

import SwiftUI

struct ProfileCard: Identifiable {
    let id = UUID()
    let imageNumber: Int
    let name: String
    let gender: String // "男" or "女"
    let age: Int
    let style: String
    let styleDescription: String
}

struct SwipeView: View {
    @State private var profiles: [ProfileCard] = [
        ProfileCard(imageNumber: 1, name: "Ivy", gender: "女", age: 26, style: "簡約風格", styleDescription: "今日選擇簡約優雅的白襯衫，搭配高腰牛仔褲，展現自信與率性。喜歡極簡主義配色，但會用小配件增添亮點。"),
        ProfileCard(imageNumber: 2, name: "Jason", gender: "男", age: 28, style: "街頭風格", styleDescription: "街頭風格混搭，寬鬆帽T與破壞牛仔，舒適又有型。熱愛街舞文化，服裝選擇偏向urban街頭風格。"),
        ProfileCard(imageNumber: 3, name: "Kevin", gender: "男", age: 30, style: "商務休閒", styleDescription: "秋冬層次穿搭，毛衣外搭長版大衣，低調卻不失質感。平日偏愛商務休閒風格，假日則會挑戰更多元素。"),
        ProfileCard(imageNumber: 4, name: "Mia", gender: "女", age: 25, style: "甜美淑女", styleDescription: "粉嫩色系連身裙搭配珍珠飾品，甜美又不失個性。喜歡嘗試不同風格，但總能保持自己的穿搭特色。")
    ]
    
    @State private var cardOffset: CGSize = .zero
    @State private var currentIndex = 0
    @State private var showExploreView = false
    @State private var selectedProfile: ProfileCard?
    
    var body: some View {
        ZStack {
            // 白色背景
            Color.white
                .ignoresSafeArea()
            
            VStack(spacing: 0) {
                // 標題
                VStack(spacing: 5) {
                    Text("穿搭配對")
                        .font(.system(size: 28, weight: .bold, design: .rounded))
                        .foregroundColor(.black)
                    
                    Text("向左滑不感興趣，向右滑收藏穿搭")
                        .font(.system(size: 14, weight: .medium, design: .rounded))
                        .foregroundColor(.gray)
                }
                .padding(.top, 20)
                .padding(.bottom, 20)
                
                // 卡片區域
                ZStack {
                    ForEach(profiles.indices.prefix(3).reversed(), id: \.self) { index in
                        CardView(
                            profile: profiles[index],
                            index: index,
                            totalCount: profiles.count,
                            offset: index == 0 ? cardOffset : .zero,
                            onSwiped: { direction in
                                // 滑動卡片後的處理邏輯
                                withAnimation(.spring()) {
                                    handleSwipe(direction: direction)
                                }
                            }
                        )
                        .offset(x: index == 0 ? cardOffset.width : 0)
                        .rotationEffect(.degrees(index == 0 ? Double(cardOffset.width / 20) : 0))
                        .scaleEffect(getScaleAmount(index: index))
                        .simultaneousGesture(
                            DragGesture()
                                .onChanged { value in
                                    if index == 0 {
                                        cardOffset = value.translation
                                    }
                                }
                                .onEnded { value in
                                    withAnimation(.spring()) {
                                        // 處理滑動結束邏輯
                                        if cardOffset.width > 120 {
                                            handleSwipe(direction: .right)
                                        } else if cardOffset.width < -120 {
                                            handleSwipe(direction: .left)
                                        } else {
                                            cardOffset = .zero
                                        }
                                    }
                                }
                        )
                    }
                }
                .padding(.top, 10)
                
                // 底部按鈕區域
                HStack(spacing: 30) {
                    Button(action: {
                        withAnimation(.spring()) {
                            handleSwipe(direction: .left)
                        }
                    }) {
                        Image(systemName: "xmark")
                            .font(.system(size: 24, weight: .bold))
                            .foregroundColor(.white)
                            .padding(15)
                            .background(
                                Circle()
                                    .fill(Color.black)
                            )
                            .shadow(color: Color.gray.opacity(0.3), radius: 5, x: 0, y: 2)
                    }
                    
                    Button(action: {
                        withAnimation(.spring()) {
                            handleSwipe(direction: .right)
                        }
                    }) {
                        Image(systemName: "heart.fill")
                            .font(.system(size: 24, weight: .bold))
                            .foregroundColor(.white)
                            .padding(15)
                            .background(
                                Circle()
                                    .fill(Color.black)
                            )
                            .shadow(color: Color.gray.opacity(0.3), radius: 5, x: 0, y: 2)
                    }
                }
                .padding(.top, 30)
                .padding(.bottom, 30)
            }
            
            if showExploreView && selectedProfile != nil {
                detailView(profile: selectedProfile!)
                    .transition(.move(edge: .bottom))
                    .zIndex(1)
            }
        }
    }
    
    // 處理滑動邏輯
    func handleSwipe(direction: SwipeDirection) {
        // 移除頂部卡片
        if !profiles.isEmpty {
            profiles.removeFirst()
        }
        
        // 重置偏移量
        cardOffset = .zero
    }
    
    // 計算卡片的縮放比例
    func getScaleAmount(index: Int) -> CGFloat {
        let cardScale: CGFloat = 0.06
        return 1.0 - (CGFloat(index) * cardScale)
    }
    
    func CardView(profile: ProfileCard, index: Int, totalCount: Int, offset: CGSize, onSwiped: @escaping (SwipeDirection) -> Void) -> some View {
        ZStack(alignment: .bottom) {
            // 卡片容器
            RoundedRectangle(cornerRadius: 25)
                .fill(Color.white)
                .frame(width: 340, height: 550)
                .shadow(color: Color.gray.opacity(0.3), radius: 5, x: 0, y: 2)
            
            // 照片 - 調整為佔據整個卡片
            Image(uiImage: PhotoHelper.loadImage(number: profile.imageNumber))
                .resizable()
                .scaledToFill()
                .frame(width: 340, height: 550)
                .clipped()
                .cornerRadius(25)
            
            // 底部半透明漸變遮罩，讓文字在照片上清晰可見
            LinearGradient(
                gradient: Gradient(colors: [Color.clear, Color.black.opacity(0.7)]),
                startPoint: .center,
                endPoint: .bottom
            )
            .frame(width: 340, height: 550)
            .cornerRadius(25)
            
            // 用戶資訊卡 - 放在底部半透明區域
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(profile.name)
                        .font(.system(size: 28, weight: .bold, design: .rounded))
                        .foregroundColor(.white)
                
                    Text("\(profile.age)")
                        .font(.system(size: 24, weight: .medium, design: .rounded))
                        .foregroundColor(.white.opacity(0.9))
                    
                    Text("・\(profile.gender)")
                        .font(.system(size: 18, weight: .medium, design: .rounded))
                        .foregroundColor(.white.opacity(0.9))
                        .padding(.leading, -5)
                }
                
                Text(profile.style)
                    .font(.system(size: 18, weight: .medium, design: .rounded))
                    .foregroundColor(.white.opacity(0.9))
                
                Text(profile.styleDescription)
                    .font(.system(size: 15, weight: .regular, design: .rounded))
                    .foregroundColor(.white.opacity(0.8))
                    .multilineTextAlignment(.leading)
                    .lineLimit(2)
                    .padding(.top, 2)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 20)
            .frame(width: 340, alignment: .leading)
        }
        .cornerRadius(25)
        .overlay(
            RoundedRectangle(cornerRadius: 25)
                .stroke(Color.gray.opacity(0.3), lineWidth: 1)
        )
    }
    
    func detailView(profile: ProfileCard) -> some View {
        ZStack {
            // 白色背景
            Color.white
                .opacity(0.95)
                .ignoresSafeArea()
                .onTapGesture {
                    withAnimation(.spring()) {
                        showExploreView = false
                    }
                }
            
            VStack {
                // 頂部標題
                Text("穿搭詳情")
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                    .foregroundColor(.black)
                    .padding(.top, 20)
                
                // 用戶詳細資料卡片
                VStack(spacing: 20) {
                    // 用戶照片
                    Image(uiImage: PhotoHelper.loadImage(number: profile.imageNumber))
                        .resizable()
                        .scaledToFill()
                        .frame(width: 150, height: 150)
                        .clipShape(Circle())
                        .overlay(
                            Circle()
                                .stroke(Color.gray.opacity(0.3), lineWidth: 2)
                        )
                        .shadow(color: Color.gray.opacity(0.2), radius: 5, x: 0, y: 2)
                    
                    // 用戶名稱和年齡
                    HStack {
                        Text(profile.name)
                            .font(.system(size: 26, weight: .bold, design: .rounded))
                            .foregroundColor(.black)
                        
                        Text("\(profile.age)")
                            .font(.system(size: 22, weight: .medium, design: .rounded))
                            .foregroundColor(.black.opacity(0.7))
                        
                        Text("・\(profile.gender)")
                            .font(.system(size: 20, weight: .medium, design: .rounded))
                            .foregroundColor(.black.opacity(0.7))
                    }
                    
                    // 風格標籤
                    Text(profile.style)
                        .font(.system(size: 20, weight: .semibold, design: .rounded))
                        .foregroundColor(.black)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 8)
                        .background(
                            Capsule()
                                .fill(Color.gray.opacity(0.1))
                                .overlay(
                                    Capsule()
                                        .stroke(Color.gray.opacity(0.3), lineWidth: 1)
                                )
                        )
                    
                    // 用戶穿搭描述
                    VStack(alignment: .leading, spacing: 10) {
                        Text("穿搭理念")
                            .font(.system(size: 20, weight: .bold, design: .rounded))
                            .foregroundColor(.black)
                        
                        Text(profile.styleDescription)
                            .font(.system(size: 17, weight: .regular, design: .rounded))
                            .foregroundColor(.gray)
                            .lineSpacing(5)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 20)
                    .padding(.top, 10)
                    
                    // 評分按鈕
                    HStack(spacing: 40) {
                        Button(action: {}) {
                            VStack {
                                Image(systemName: "star.fill")
                                    .font(.system(size: 22))
                                    .foregroundColor(.white)
                                Text("給予評分")
                                    .font(.system(size: 12, weight: .medium))
                                    .foregroundColor(.white)
                            }
                            .padding(.vertical, 15)
                            .padding(.horizontal, 20)
                            .background(
                                RoundedRectangle(cornerRadius: 12)
                                    .fill(Color.black)
                            )
                            .shadow(color: Color.gray.opacity(0.2), radius: 5, x: 0, y: 2)
                        }
                        
                        Button(action: {}) {
                            VStack {
                                Image(systemName: "message.fill")
                                    .font(.system(size: 22))
                                    .foregroundColor(.white)
                                Text("穿搭建議")
                                    .font(.system(size: 12, weight: .medium))
                                    .foregroundColor(.white)
                            }
                            .padding(.vertical, 15)
                            .padding(.horizontal, 20)
                            .background(
                                RoundedRectangle(cornerRadius: 12)
                                    .fill(Color.black)
                            )
                            .shadow(color: Color.gray.opacity(0.2), radius: 5, x: 0, y: 2)
                        }
                    }
                    .padding(.top, 20)
                }
                .padding(.vertical, 30)
                .padding(.horizontal, 20)
                .background(Color.white)
                .cornerRadius(30)
                .shadow(color: Color.gray.opacity(0.2), radius: 10, x: 0, y: 5)
                .overlay(
                    RoundedRectangle(cornerRadius: 30)
                        .stroke(Color.gray.opacity(0.3), lineWidth: 1)
                )
                .padding(.horizontal, 15)
                
                // 關閉按鈕
                Button(action: {
                    withAnimation(.spring()) {
                        showExploreView = false
                    }
                }) {
                    Text("關閉")
                        .font(.system(size: 18, weight: .medium, design: .rounded))
                        .foregroundColor(.white)
                        .padding(.vertical, 12)
                        .padding(.horizontal, 40)
                        .background(
                            Capsule()
                                .fill(Color.black)
                                .shadow(color: Color.gray.opacity(0.2), radius: 5, x: 0, y: 2)
                        )
                }
                .padding(.top, 25)
                
                Spacer()
            }
            .padding()
        }
    }
}

// 滑動方向枚舉
enum SwipeDirection {
    case left
    case right
}

// 按鈕彈性效果
struct BouncyButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.85 : 1)
            .animation(.spring(), value: configuration.isPressed)
    }
}

// 圓角擴展
struct RoundedCorner: Shape {
    var radius: CGFloat = .infinity
    var corners: UIRectCorner = .allCorners

    func path(in rect: CGRect) -> Path {
        let path = UIBezierPath(roundedRect: rect, byRoundingCorners: corners, cornerRadii: CGSize(width: radius, height: radius))
        return Path(path.cgPath)
    }
}

extension View {
    func cornerRadius(_ radius: CGFloat, corners: UIRectCorner) -> some View {
        clipShape(RoundedCorner(radius: radius, corners: corners))
    }
}

struct SwipeView_Previews: PreviewProvider {
    static var previews: some View {
        SwipeView()
    }
} 