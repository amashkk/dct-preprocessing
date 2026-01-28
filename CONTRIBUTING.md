## 開始之前

1. 在 GitHub 上 **Fork** 這個專案
2. **Clone** 到本地端：
   ```bash
   git clone https://github.com/你的帳號/dct-preprocessing.git
   cd dct-preprocessing
   ```
3. **安裝相依套件**：
   ```bash
   pip install -r requirements.txt
   ```

## 開發環境設定

### 執行測試

```bash
python -m pytest tests/
```

### 程式碼風格

我們遵循 PEP 8 風格指南。請確保您的程式碼：
- 使用 4 個空格進行縮排
- 為所有公開函式和類別撰寫 docstring
- 使用有意義的變數名稱

## 如何貢獻

### 回報錯誤

如果您發現錯誤，請建立 issue 並包含：
- 清楚、描述性的標題
- 重現錯誤的步驟
- 預期行為 vs 實際行為
- 您的環境（Python 版本、作業系統等）

### 建議新功能

對於功能請求：
- 開啟一個 issue 描述該功能
- 解釋為什麼這個功能會有用
- 如果可能，提供範例

### Pull Requests

1. 為您的功能建立新分支：
   ```bash
   git checkout -b feature/功能名稱
   ```

2. 進行修改並提交：
   ```bash
   git add .
   git commit -m "Add: 簡短描述修改內容"
   ```

3. Push 到您的 fork：
   ```bash
   git push origin feature/功能名稱
   ```

4. 在 GitHub 上開啟 Pull Request

### Commit 訊息指南

使用清楚、描述性的 commit 訊息：
- `Add:` 新增功能
- `Fix:` 修復錯誤
- `Update:` 更新現有功能
- `Docs:` 文件修改
- `Refactor:` 程式碼重構
