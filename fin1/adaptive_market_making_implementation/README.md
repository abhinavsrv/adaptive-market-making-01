# Adaptive Market Making Implementation

This repository contains a complete implementation of an Adaptive Market Making Strategy with Deep Learning-Based Regime Detection and Game-Theoretic Strategy for High-Frequency Futures Trading.

## Overview

This project implements a sophisticated market making strategy that adapts to changing market conditions through:

1. **Deep Learning-Based Regime Detection** - Using autoencoders to identify market regimes
2. **Bayesian Estimation of Informed Trading** - Detecting adverse selection risk
3. **Game-Theoretic Optimal Spread Calculation** - Adapting spreads based on market conditions
4. **Real-Time Execution Engine** - Handling order placement and risk management
5. **Comprehensive Backtesting Framework** - For strategy validation and optimization

The system is designed for high-frequency futures trading, with particular focus on CME Globex markets.

## System Architecture

The implementation follows a modular architecture with the following components:

```
adaptive_market_making_implementation/
├── src/
│   ├── data_ingestion/        # Market data collection and processing
│   ├── feature_engineering/   # Feature extraction and transformation
│   ├── regime_detection/      # Autoencoder and Bayesian models
│   ├── strategy/              # Adaptive market making strategy
│   ├── execution/             # Order execution and risk management
│   ├── backtesting/           # Backtesting and simulation framework
│   └── utils/                 # Utility functions and helpers
├── config/                    # Configuration files
├── data/                      # Data storage
├── models/                    # Trained models
├── tests/                     # Unit and integration tests
└── docs/                      # Documentation
```

## Key Features

- **Adaptive Spread Calculation**: Dynamically adjusts bid-ask spreads based on market regime, volatility, and inventory
- **Market Regime Detection**: Uses deep learning to identify different market states and adapt strategy accordingly
- **Informed Trading Detection**: Bayesian estimation of adverse selection risk to protect against informed traders
- **Inventory Management**: Optimal inventory control to balance position risk and trading opportunities
- **Real-Time Processing**: Kafka-based data pipeline for low-latency market data processing
- **Time-Series Storage**: InfluxDB for efficient storage and querying of market data
- **Stream Processing**: Apache Flink for complex event processing and feature engineering
- **Multi-Agent Simulation**: Realistic market simulation with different agent types for strategy testing

## Requirements

- Python 3.8+
- Kafka
- InfluxDB
- Apache Flink
- PyTorch
- NumPy, Pandas, Scikit-learn
- Confluent Kafka Python client
- InfluxDB Python client

## Installation

1. Clone the repository:
```bash
git clone https://github.com/abhinavsrv/adaptive-market-making.git
cd adaptive-market-making
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up infrastructure components:
```bash
# Start Kafka
docker-compose -f docker/kafka.yml up -d

# Start InfluxDB
docker-compose -f docker/influxdb.yml up -d

# Start Flink
docker-compose -f docker/flink.yml up -d
```

4. Configure the system:
```bash
# Edit configuration files in the config/ directory
nano config/data_sources.yaml
nano config/model_params.yaml
nano config/strategy_params.yaml
```

## Usage

### Running the System

To start the system in live trading mode:

```bash
python src/main.py --config config/config.yaml --mode live
```

### Backtesting

To run a backtest on historical data:

```bash
python src/main.py --config config/config.yaml --mode backtest --data data/historical/es_futures_2023.csv --output results/backtest_20230101
```

### Simulation

To run a multi-agent simulation:

```bash
python src/main.py --config config/config.yaml --mode simulation --output results/simulation_001
```

## For Quantitative Researchers

The system is designed to be easily extensible for research purposes:

- Implement new regime detection models in `src/regime_detection/`
- Experiment with different spread calculation algorithms in `src/strategy/`
- Add new features to the feature engineering pipeline in `src/feature_engineering/`
- Analyze backtest results and optimize parameters

See `docs/researcher_guide.md` for detailed information.

## For Traders

The system provides a robust framework for live trading:

- Monitor strategy performance in real-time
- Adjust risk parameters on-the-fly
- Implement circuit breakers and safety mechanisms
- Integrate with existing trading infrastructure

See `docs/trader_guide.md` for detailed information.

## License

This project is proprietary and confidential. All rights reserved.

## Author

Abhinav Srivastava
