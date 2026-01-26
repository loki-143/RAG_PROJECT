# Azure App Service Deployment Guide

## 1. Create Azure Resources

```bash
# Login to Azure
az login

# Create resource group
az group create --name rag-project-rg --location eastus

# Deploy using template
az deployment group create \
  --resource-group rag-project-rg \
  --template-file azure-deploy.json \
  --parameters siteName=rag-agent-api
```

## 2. Configure in Azure Portal

1. Go to [portal.azure.com](https://portal.azure.com)
2. Find your App Service: **rag-agent-api**
3. **Settings → Configuration → Application settings**
4. Add these:
   - `GOOGLE_API_KEY` = your key
   - `RAG_API_KEY` = test123
   - `ENVIRONMENT` = production
   - `ENABLE_DOCS` = true
   - `SCM_DO_BUILD_DURING_DEPLOYMENT` = true

## 3. Setup GitHub Deployment

1. **Deployment Center**
2. Source: **GitHub**
3. Connect repo and select `RAG_PROJECT`, branch `main`
4. **Save**

## 4. Check Deployment

Your app will be at:
```
https://rag-agent-api.azurewebsites.net
```

---

## Free Tier Limits
- 60 min/day compute time
- 1 GB RAM
- For more, upgrade to B1 (~$13/mo, use student credit)
