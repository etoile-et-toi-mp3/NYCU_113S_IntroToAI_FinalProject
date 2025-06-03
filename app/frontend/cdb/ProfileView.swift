//
//  ProfileView.swift
//  cdb
//
//  Created by kevin Chou on 2025/4/28.
//

import SwiftUI

struct UserProfile {
    var username: String
    var gender: String // "男" or "女"
    var bio: String
    var profileImageNumber: Int
    var posts: Int
    var followers: Int
    var following: Int
    var outfits: [UserOutfit]
    var style: String
}

struct UserOutfit: Identifiable {
    let id = UUID()
    let imageNumber: Int
    let likes: Int
    let description: String
}

struct ProfileView: View {
    @State private var isEditing = false
    @State private var name = "王小明"
    @State private var bio = "熱愛時尚的設計學生，喜歡探索不同風格的穿搭，尋找屬於自己的獨特風格。目前就讀台北設計學院，專攻服裝設計。"
    @State private var selectedInterests = ["街頭風格", "簡約風格", "日系"]
    @State private var showingSettings = false
    
    let interests = ["街頭風格", "簡約風格", "日系", "韓系", "歐美", "運動休閒", "正裝", "古著", "龐克", "嘻哈"]
    
    var body: some View {
        ScrollView {
            VStack(spacing: 25) {
                // 用戶資料區
                VStack(spacing: 15) {
                    // 頭像和數據卡片
                    VStack {
                        // 頭像
                        ZStack {
                            if UIImage(named: "avatar") != nil {
                                Image("avatar")
                                    .resizable()
                                    .scaledToFill()
                                    .frame(width: 100, height: 100)
                                    .clipShape(Circle())
                            } else {
                                Image(systemName: "person.crop.circle.fill")
                                    .font(.system(size: 100))
                                    .foregroundColor(.gray.opacity(0.6))
                                    .frame(width: 100, height: 100)
                            }
                        }
                        .overlay(Circle().stroke(Color.gray.opacity(0.3), lineWidth: 1))
                        .shadow(color: Color.gray.opacity(0.2), radius: 4)
                        
                        // 用戶信息
                        if isEditing {
                            editingView
                        } else {
                            displayView
                        }
                    }
                    .padding(.horizontal, 20)
                    
                    // 數據卡片
                    HStack(spacing: 0) {
                        dataCard(title: "穿搭", value: "43")
                        
                        Divider()
                            .frame(height: 40)
                        
                        dataCard(title: "粉絲", value: "512")
                        
                        Divider()
                            .frame(height: 40)
                        
                        dataCard(title: "按讚", value: "1.2k")
                    }
                    .frame(height: 80)
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color.white)
                            .shadow(color: Color.gray.opacity(0.15), radius: 5, x: 0, y: 2)
                    )
                    .padding(.horizontal, 20)
                }
                .padding(.top, 20)
                
                // 穿搭圖片集合
                VStack(alignment: .leading, spacing: 15) {
                    Text("我的穿搭")
                        .font(.system(size: 20, weight: .bold, design: .rounded))
                        .padding(.horizontal, 20)
                    
                    outfitGridView
                }
                
                Spacer(minLength: 50)
            }
            .padding(.top, 10)
        }
        .background(Color.white.ignoresSafeArea())
        .navigationBarTitle("個人資料", displayMode: .inline)
        .navigationBarItems(
            trailing: HStack(spacing: 15) {
                // 設定按鈕
                Button(action: {
                    showingSettings = true
                }) {
                    Image(systemName: "gearshape.fill")
                        .font(.system(size: 20))
                        .foregroundColor(.black)
                }
                
                // 編輯按鈕
                editButton
            }
        )
        .sheet(isPresented: $showingSettings) {
            SettingsView()
        }
    }
    
    // 資料卡片
    private func dataCard(title: String, value: String) -> some View {
        VStack(spacing: 5) {
            Text(value)
                .font(.system(size: 22, weight: .bold))
                .foregroundColor(.black)
            
            Text(title)
                .font(.system(size: 14))
                .foregroundColor(.gray)
        }
        .frame(maxWidth: .infinity)
    }
    
    // 編輯按鈕
    private var editButton: some View {
        Button(action: {
            withAnimation {
                isEditing.toggle()
            }
        }) {
            Text(isEditing ? "完成" : "編輯")
                .font(.system(size: 16, weight: .medium))
                .foregroundColor(.black)
        }
    }
    
    // 顯示模式視圖
    private var displayView: some View {
        VStack(spacing: 12) {
            Text(name)
                .font(.system(size: 24, weight: .bold, design: .rounded))
                .foregroundColor(.black)
            
            Text(bio)
                .font(.system(size: 14, weight: .regular, design: .rounded))
                .foregroundColor(.gray)
                .multilineTextAlignment(.center)
                .lineSpacing(4)
                .padding(.horizontal, 20)
            
            // 興趣標籤
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(selectedInterests, id: \.self) { interest in
                        Text(interest)
                            .font(.system(size: 13, weight: .medium))
                            .foregroundColor(.black)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(
                                RoundedRectangle(cornerRadius: 16)
                                    .fill(Color.gray.opacity(0.1))
                            )
                    }
                }
                .padding(.horizontal, 10)
            }
        }
        .padding(.top, 10)
        .padding(.bottom, 15)
    }
    
    // 編輯模式視圖
    private var editingView: some View {
        VStack(spacing: 15) {
            TextField("名稱", text: $name)
                .font(.system(size: 20, weight: .bold))
                .multilineTextAlignment(.center)
                .padding(.horizontal, 20)
            
            ZStack(alignment: .topLeading) {
                if bio.isEmpty {
                    Text("個人簡介...")
                        .font(.system(size: 14))
                        .foregroundColor(.gray.opacity(0.8))
                        .padding(.horizontal, 25)
                        .padding(.top, 8)
                }
                
                TextEditor(text: $bio)
                    .font(.system(size: 14))
                    .foregroundColor(.gray)
                    .frame(height: 80)
                    .padding(.horizontal, 20)
                    .background(Color.gray.opacity(0.05))
                    .cornerRadius(8)
            }
            .frame(height: 80)
            
            // 興趣選擇網格
            VStack(alignment: .leading, spacing: 10) {
                Text("興趣")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(.black)
                
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 100), spacing: 10)], spacing: 10) {
                    ForEach(interests, id: \.self) { interest in
                        interestToggle(interest)
                    }
                }
            }
            .padding(.horizontal, 20)
        }
        .padding(.vertical, 15)
    }
    
    // 興趣標籤切換
    private func interestToggle(_ interest: String) -> some View {
        let isSelected = selectedInterests.contains(interest)
        
        return Button(action: {
            withAnimation {
                if isSelected {
                    selectedInterests.removeAll { $0 == interest }
                } else {
                    selectedInterests.append(interest)
                }
            }
        }) {
            Text(interest)
                .font(.system(size: 13, weight: .medium))
                .foregroundColor(isSelected ? .white : .black)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(
                    RoundedRectangle(cornerRadius: 16)
                        .fill(isSelected ? Color.black : Color.gray.opacity(0.1))
                )
        }
    }
    
    // 穿搭網格視圖
    private var outfitGridView: some View {
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())], spacing: 3) {
            ForEach(1...12, id: \.self) { index in
                Image("outfit\(index % 4 + 1)")
                    .resizable()
                    .scaledToFill()
                    .frame(height: 120)
                    .clipped()
                    .contentShape(Rectangle())
            }
        }
    }
}

struct ProfileView_Previews: PreviewProvider {
    static var previews: some View {
        ProfileView()
    }
} 