param(
    [string]$PythonExe = "python",
    [int[]]$Seeds = @(42, 43, 44, 45, 46),
    [int]$Epochs = 50,
    [int]$BatchSize = 128,
    [Parameter(Mandatory=$true)][string]$TrainCsv,
    [Parameter(Mandatory=$true)][string]$ResNetTokens,
    [Parameter(Mandatory=$true)][string]$ResNetGlobal,
    [string]$EfficientNetFeatures = "",
    [string]$VitFeatures = "",
    [string]$ResultsRoot = "./results/fair_e1_e7",
    [switch]$AllowFeatureProxy
)

$ErrorActionPreference = "Stop"
Get-Command -Name $PythonExe -CommandType Application -ErrorAction Stop | Out-Null
$EntryPoint = Join-Path $PSScriptRoot "main.py"
if (-not (Test-Path $TrainCsv)) {
    throw "CSV not found: $TrainCsv"
}
if (-not (Test-Path $ResNetTokens)) {
    throw "ResNet token features not found: $ResNetTokens"
}
if (-not (Test-Path $ResNetGlobal)) {
    throw "ResNet global features not found: $ResNetGlobal"
}

if (-not $AllowFeatureProxy) {
    if ([string]::IsNullOrWhiteSpace($EfficientNetFeatures) -or -not (Test-Path $EfficientNetFeatures)) {
        throw "Exact E2 requires an existing -EfficientNetFeatures .npy file. Missing: $EfficientNetFeatures. Generate it with extract_rgb_backbone_features.py or use -AllowFeatureProxy only for smoke/proxy validation."
    }
    if ([string]::IsNullOrWhiteSpace($VitFeatures) -or -not (Test-Path $VitFeatures)) {
        throw "Exact E3 requires an existing -VitFeatures .npy file. Missing: $VitFeatures. Generate it with extract_rgb_backbone_features.py or use -AllowFeatureProxy only for smoke/proxy validation."
    }
}

foreach ($seed in $Seeds) {
    $outputDir = Join-Path $ResultsRoot "seed_$seed"
    $argsList = @(
        $EntryPoint,
        "--study_name", "fair_e1_e7",
        "--mode", "train",
        "--seed", "$seed",
        "--epochs", "$Epochs",
        "--batch_size", "$BatchSize",
        "--train_csv_path", $TrainCsv,
        "--image_features_path", $ResNetTokens,
        "--global_image_features_path", $ResNetGlobal,
        "--token_image_features_path", $ResNetTokens,
        "--output_dir", $outputDir
    )
    if (-not [string]::IsNullOrWhiteSpace($EfficientNetFeatures)) {
        $argsList += @("--efficientnet_features_path", $EfficientNetFeatures)
    }
    if (-not [string]::IsNullOrWhiteSpace($VitFeatures)) {
        $argsList += @("--vit_features_path", $VitFeatures)
    }
    if ($AllowFeatureProxy) {
        $argsList += "--allow_feature_proxy"
    }

    & $PythonExe @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "fair_e1_e7 failed for seed $seed with exit code $LASTEXITCODE"
    }
}
