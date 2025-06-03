//
//  ImageCache.swift
//  cdb
//
//  Created by AI Assistant on 2025/1/1.
//

import SwiftUI
import Foundation

// MARK: - 圖片快取管理器
class ImageCache: ObservableObject {
    static let shared = ImageCache()
    private var cache = NSCache<NSString, UIImage>()
    
    private init() {
        cache.countLimit = 100 // 限制快取數量
        cache.totalCostLimit = 50 * 1024 * 1024 // 限制快取大小為50MB
    }
    
    func getImage(for url: URL) -> UIImage? {
        return cache.object(forKey: url.absoluteString as NSString)
    }
    
    func setImage(_ image: UIImage, for url: URL) {
        cache.setObject(image, forKey: url.absoluteString as NSString)
    }
    
    func clearCache() {
        cache.removeAllObjects()
    }
}

// MARK: - 快取圖片視圖
struct CachedAsyncImage<Content: View>: View {
    let url: URL?
    let content: (AsyncImagePhase) -> Content
    
    @StateObject private var imageCache = ImageCache.shared
    @State private var phase: AsyncImagePhase = .empty
    @State private var imageTask: Task<Void, Never>?
    
    init(url: URL?, @ViewBuilder content: @escaping (AsyncImagePhase) -> Content) {
        self.url = url
        self.content = content
    }
    
    var body: some View {
        content(phase)
            .onAppear {
                loadImage()
            }
            .onDisappear {
                imageTask?.cancel()
            }
    }
    
    private func loadImage() {
        guard let url = url else {
            phase = .failure(URLError(.badURL))
            return
        }
        
        // 檢查快取
        if let cachedImage = imageCache.getImage(for: url) {
            phase = .success(Image(uiImage: cachedImage))
            return
        }
        
        // 開始載入
        phase = .empty
        
        imageTask = Task {
            do {
                let (data, _) = try await URLSession.shared.data(from: url)
                
                guard !Task.isCancelled else { return }
                
                if let uiImage = UIImage(data: data) {
                    // 儲存到快取
                    await MainActor.run {
                        imageCache.setImage(uiImage, for: url)
                        phase = .success(Image(uiImage: uiImage))
                    }
                } else {
                    await MainActor.run {
                        phase = .failure(URLError(.cannotDecodeContentData))
                    }
                }
            } catch {
                guard !Task.isCancelled else { return }
                await MainActor.run {
                    phase = .failure(error)
                }
            }
        }
    }
} 