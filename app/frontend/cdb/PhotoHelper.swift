//
//  PhotoHelper.swift
//  cdb
//
//  Created by kevin Chou on 2025/4/28.
//

import SwiftUI

struct PhotoHelper {
    static func loadImage(number: Int) -> UIImage {
        // 首先嘗試從Bundle中加載
        if let image = UIImage(named: "\(number)") {
            return image
        }
        
        // 嘗試從照片目錄加載
        let filePath = Bundle.main.path(forResource: "\(number)", ofType: "jpg", inDirectory: "photos")
        if let filePath = filePath, let image = UIImage(contentsOfFile: filePath) {
            return image
        }
        
        // 嘗試從應用文檔目錄加載
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let photosPath = documentsPath.appendingPathComponent("photos")
        let imagePath = photosPath.appendingPathComponent("\(number).jpg")
        
        if let image = UIImage(contentsOfFile: imagePath.path) {
            return image
        }
        
        // 最後嘗試從工作目錄加載
        let workingDirPath = "../photos/\(number).jpg"
        if let image = UIImage(contentsOfFile: workingDirPath) {
            return image
        }
        
        // 如果所有嘗試都失敗，返回預設圖片
        return UIImage(systemName: "person.fill") ?? UIImage()
    }
    
    // 獲取所有照片數量
    static func getPhotoCount() -> Int {
        // 這裡我們通過嘗試加載照片從1開始，直到找不到照片為止
        var count = 0
        for i in 1...100 {
            let testImage = loadImage(number: i)
            if testImage.size.width > 0 {
                count = i
            } else {
                break
            }
        }
        return count > 0 ? count : 4  // 至少返回4張照片（因為我們知道有1-4.jpg）
    }
} 