//
//  ContentView.swift
//  cdb
//
//  Created by kevin Chou on 2025/4/28.
//

import SwiftUI

struct ContentView: View {
    @State private var selectedTab = 0 // 默認選中探索頁面
    @State private var showingCameraView = false // 控制相機界面的顯示
    
    var body: some View {
        ZStack {
            // 主要的 TabView
            TabView(selection: $selectedTab) {
                // 探索頁面 - SwipeView
                SwipeView()
                    .tabItem {
                        Image(systemName: selectedTab == 0 ? "heart.fill" : "heart")
                        Text("探索")
                    }
                    .tag(0)
                
                // 分析頁面 - 空白視圖，實際功能由 fullScreenCover 處理
                Color.clear
                    .tabItem {
                        Image(systemName: selectedTab == 1 ? "camera.fill" : "camera")
                        Text("分析")
                    }
                    .tag(1)
                
                // 個人頁面 - 包裝在 NavigationView 中
                NavigationView {
                    ProfileView()
                }
                .tabItem {
                    Image(systemName: selectedTab == 2 ? "person.fill" : "person")
                    Text("個人")
                }
                .tag(2)
            }
            .accentColor(.black) // 設置選中時的顏色
            .preferredColorScheme(.light) // 使用淺色主題以配合整體設計
        }
        .onChange(of: selectedTab) { newValue in
            if newValue == 1 {
                showingCameraView = true
            }
        }
        .fullScreenCover(isPresented: $showingCameraView) {
            CameraView(onDismiss: {
                showingCameraView = false
                selectedTab = 0 // 返回探索頁面
            })
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
} 