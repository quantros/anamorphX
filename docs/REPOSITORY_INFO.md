# 📦 Repository Information

## 🔗 GitHub Repository
**URL**: https://github.com/quantros/anamorphX

## 🌿 Branch Structure

### Main Branches
- **`main`** - Production-ready code, stable releases
- **`develop`** - Integration branch for ongoing development

### Feature Branches
- **`feature/lexer-parser`** - Lexical analyzer and parser implementation
- **`feature/signal-system`** - Unique signal processing system
- **`feature/rest-api`** - FastAPI-based REST API and web interface

## 📋 GitHub Issues Created

1. **#1** - 🔤 Implement Lexer with Case-Insensitive Keywords
   - Priority: High, Milestone: Stage 2
   - 80 neural commands tokenization, async support

2. **#2** - 🔧 Implement Parser with Error Recovery  
   - Priority: High, Milestone: Stage 2
   - Recursive descent parser, 4 error recovery strategies

3. **#3** - ⚡ Implement Signal Processing System
   - Priority: High, Milestone: Stage 3
   - 4 signal types, queues, retry logic, monitoring

4. **#4** - 🌐 Implement REST API with OpenAPI
   - Priority: Medium, Milestone: Stage 4
   - FastAPI, authentication, plugin system

## 🚀 Development Workflow

### Git Flow
1. **Feature Development**: Create feature branch from `develop`
2. **Pull Requests**: Merge feature branches into `develop`
3. **Releases**: Merge `develop` into `main` for releases
4. **Hotfixes**: Create hotfix branches from `main`

### Commit Convention
```
<type>(<scope>): <description>

Types: feat, fix, docs, style, refactor, test, chore
Scopes: lexer, parser, signals, api, security, monitoring
```

### Example Commits
```bash
feat(lexer): implement case-insensitive keyword tokenization
fix(parser): resolve error recovery in nested expressions  
docs(api): update OpenAPI specification
test(signals): add fuzz testing for signal processing
```

## 🔧 Local Development Setup

```bash
# Clone repository
git clone https://github.com/quantros/anamorphX.git
cd anamorphX

# Switch to develop branch
git checkout develop

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Start development server (when implemented)
uvicorn src.api.main:app --reload
```

## 📊 Project Status

- **Repository**: ✅ Created and configured
- **Branches**: ✅ Main development branches set up
- **Issues**: ✅ Initial development tasks created
- **Documentation**: ✅ Comprehensive docs available
- **Architecture**: ✅ Enterprise-ready design complete

## 🎯 Next Steps

1. **Start with Issue #1**: Implement lexer on `feature/lexer-parser` branch
2. **Parallel development**: Begin REST API on `feature/rest-api` branch  
3. **Testing**: Set up CI/CD pipeline with GitHub Actions
4. **Documentation**: Create GitHub Wiki for detailed guides

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'feat: add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request to `develop` branch

## 📞 Repository Management

- **Owner**: quantros
- **Visibility**: Public
- **License**: MIT
- **Language**: Python
- **Topics**: programming-language, neural-networks, security, enterprise

---

**🧠 Ready to start coding the future of neural programming! 🚀**
