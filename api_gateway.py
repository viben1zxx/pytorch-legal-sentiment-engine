{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/viben1zxx/pytorch-legal-sentiment-engine/blob/main/api_gateway.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 128,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "92twBrcuaT_s",
        "outputId": "861a9bb3-99df-429e-f57c-46f45bf1e01d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "2.9.0+cu126\n",
            "GPU is available!\n",
            "Using GPU:Tesla T4\n"
          ]
        }
      ],
      "source": [
        "import torch\n",
        "print(torch.__version__)\n",
        "\n",
        "if torch.cuda.is_available():\n",
        "    print(\"GPU is available!\")\n",
        "    print(f\"Using GPU:{torch.cuda.get_device_name(0)}\")\n",
        "else :\n",
        "      print(\"GPU not available. Using cpu.\")\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 129,
      "metadata": {
        "id": "gByEnWywgUyL"
      },
      "outputs": [],
      "source": [
        "a=torch.empty(2,3)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 130,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "JKioR6WoghAm",
        "outputId": "78a3d530-27f5-4c63-c320-9d1f2209d1cf"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "torch.Tensor"
            ]
          },
          "metadata": {},
          "execution_count": 130
        }
      ],
      "source": [
        "type(a)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 131,
      "metadata": {
        "id": "-anvWkNKf6gk"
      },
      "outputs": [],
      "source": [
        "def dy_dx(x) :\n",
        "  return 2*x\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 132,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "oBRqXTP4gVMz",
        "outputId": "5c79d07c-2142-4864-cfee-f4cb516f8f1a"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "6"
            ]
          },
          "metadata": {},
          "execution_count": 132
        }
      ],
      "source": [
        "dy_dx(3)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 133,
      "metadata": {
        "id": "pbsT5zEwgj9e"
      },
      "outputs": [],
      "source": [
        "import math\n",
        "\n",
        "def dz_dx(x) :\n",
        "  return 2 * x * math.cos(x**2)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 134,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Bh91gkbXhDhZ",
        "outputId": "a6a2a03e-14f6-4ffc-b4c5-d7a583c77daa"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "-5.466781571308061"
            ]
          },
          "metadata": {},
          "execution_count": 134
        }
      ],
      "source": [
        "dz_dx(3)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 135,
      "metadata": {
        "id": "kvW0Am-9ha-l"
      },
      "outputs": [],
      "source": [
        "def dz_dy(x) :\n",
        "  return 2 * x * math.cos(x**2)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 136,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "T9XlFThGiJjA",
        "outputId": "3b4fbd4f-1076-4f7e-9e1b-79d830b47e0c"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "-7.661275842587077"
            ]
          },
          "metadata": {},
          "execution_count": 136
        }
      ],
      "source": [
        "dz_dx(4)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 137,
      "metadata": {
        "id": "jUy6y_GYmqn2"
      },
      "outputs": [],
      "source": [
        "import torch"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 138,
      "metadata": {
        "id": "9b2hij-eoElh"
      },
      "outputs": [],
      "source": [
        "x = torch.tensor(4.0, requires_grad=True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 139,
      "metadata": {
        "id": "6wfYk37voVEX"
      },
      "outputs": [],
      "source": [
        "y = x**2\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 140,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "oLZbqx-Zo5tj",
        "outputId": "432b0443-64fa-4030-e16b-1b5d1ae17046"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "tensor(4., requires_grad=True)"
            ]
          },
          "metadata": {},
          "execution_count": 140
        }
      ],
      "source": [
        "x"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 141,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yQwyv26lohwF",
        "outputId": "0f9bb1b5-0cc8-4d02-abc1-8324a9fb1aac"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "tensor(16., grad_fn=<PowBackward0>)"
            ]
          },
          "metadata": {},
          "execution_count": 141
        }
      ],
      "source": [
        "y"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 142,
      "metadata": {
        "id": "pkWY5RAtppJ1"
      },
      "outputs": [],
      "source": [
        "y.backward()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 143,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OHixbi5ZpwuA",
        "outputId": "a042c071-c3fb-4d50-c686-b176e867ce8a"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "tensor(8.)"
            ]
          },
          "metadata": {},
          "execution_count": 143
        }
      ],
      "source": [
        "x.grad"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 144,
      "metadata": {
        "id": "RRfSOKrMqizO"
      },
      "outputs": [],
      "source": [
        "x = torch.tensor(4.0, requires_grad=True)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 145,
      "metadata": {
        "id": "0HY6TWqhrAhC"
      },
      "outputs": [],
      "source": [
        "y = x** 2"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 146,
      "metadata": {
        "id": "uD9m8p1FrHqb"
      },
      "outputs": [],
      "source": [
        "z = torch.sin(y)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 147,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FtMLNr8LrN7J",
        "outputId": "d7f2a273-c76e-41be-dc5e-3429b6c284c5"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "tensor(4., requires_grad=True)"
            ]
          },
          "metadata": {},
          "execution_count": 147
        }
      ],
      "source": [
        "x\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 148,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9--w_HA8rQBS",
        "outputId": "36b17462-064e-40a0-bee7-3830b4707ce7"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "tensor(16., grad_fn=<PowBackward0>)"
            ]
          },
          "metadata": {},
          "execution_count": 148
        }
      ],
      "source": [
        "y"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 149,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7xSlWqUqrSPs",
        "outputId": "747cf0c3-43b6-49b2-ce13-3ab9878e0a30"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "tensor(-0.2879, grad_fn=<SinBackward0>)"
            ]
          },
          "metadata": {},
          "execution_count": 149
        }
      ],
      "source": [
        "z"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 150,
      "metadata": {
        "id": "CY6mrfYmt2gW"
      },
      "outputs": [],
      "source": [
        "z.backward()\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 151,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mvCoiFKPt_Ek",
        "outputId": "f405f5b0-d7fd-4044-91c7-cca8b0726d24"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "tensor(-7.6613)"
            ]
          },
          "metadata": {},
          "execution_count": 151
        }
      ],
      "source": [
        "x.grad"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 152,
      "metadata": {
        "id": "3wWjc22ru1Zp"
      },
      "outputs": [],
      "source": [
        "import torch\n",
        "x = torch.tensor(6.7) #input figure\n",
        "y = torch.tensor(0.0) #true label (binary)\n",
        "\n",
        "w = torch.tensor(1.0) #weight\n",
        "b = torch.tensor(0.0) #Bias\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 153,
      "metadata": {
        "id": "9yIA8iE2wcD-"
      },
      "outputs": [],
      "source": [
        "# Binary Cross-Entropy Loss for scalar\n",
        "def binary_cross_entropy_loss(prediction, target):\n",
        "  epsilon = 1e-8 # To prevent log(0)\n",
        "  predction = torch.clamp(prediction, epsilon,1 - epsilon)\n",
        "  return -(target * torch.log(prediction) + (1 - target) * torch.log(1 - prediction))\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 154,
      "metadata": {
        "id": "x_mdY6K_2FGL"
      },
      "outputs": [],
      "source": [
        "# Forward pass\n",
        "z = w * x + b # weighted sum (linear part)\n",
        "y_pred = torch.sigmoid(z) # predicted probability\n",
        "\n",
        "# compute binary cross-entropy loss\n",
        "loss = binary_cross_entropy_loss(y_pred, y)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 155,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-XBMxv8R-Yut",
        "outputId": "762eaeb5-22f0-41c8-9ffd-9159e9b36b19"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "tensor(6.7012)"
            ]
          },
          "metadata": {},
          "execution_count": 155
        }
      ],
      "source": [
        "loss"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 156,
      "metadata": {
        "id": "bJ2nst4V3prP"
      },
      "outputs": [],
      "source": [
        "\n",
        " # Derivatives:\n",
        "#1. dL/d(y_pred): Loss with respect to the prediction (y_pred)\n",
        "dloss_dy_pred = (y_pred - y)/(y_pred*(1-y_pred))\n",
        "# 2. dy_pred/dz: Prediction (y_pred) with respect to z (sigmoid derivative)\n",
        "dy_pred_dz = y_pred * (1 - y_pred)\n",
        "\n",
        "# 3. dz/dw and dz/db: z with respect to w and b\n",
        "dZ_dw = x  #dz/dw = x\n",
        "dZ_db = 1  #dz/db = 1 (bias contributes directly to z)\n",
        "\n",
        "\n",
        "dL_dw = dloss_dy_pred * dy_pred_dz * dZ_dw\n",
        "dL_db = dloss_dy_pred * dy_pred_dz * dZ_db\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 157,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "D1kFVtd2AM7q",
        "outputId": "c8663b7c-e4cb-404e-e28c-f86d3f53f1f1"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "manual gradient of loss w.r.t weight (dw): 6.691762447357178\n",
            "manaul gradient of loss w.r.t bias (db): 0.998770534992218\n"
          ]
        }
      ],
      "source": [
        "print(f\"manual gradient of loss w.r.t weight (dw): {dL_dw}\")\n",
        "print(f\"manaul gradient of loss w.r.t bias (db): {dL_db}\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 158,
      "metadata": {
        "id": "-HuO8vEUBgUa"
      },
      "outputs": [],
      "source": [
        "x = torch.tensor(6.7)\n",
        "y = torch.tensor(0.0)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 159,
      "metadata": {
        "id": "IgjgKftJB0oH"
      },
      "outputs": [],
      "source": [
        "w = torch.tensor(1.0, requires_grad=True)\n",
        "b = torch.tensor(0.0, requires_grad=True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 160,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aoqV0VW6CesD",
        "outputId": "a3f985b7-07e6-49a7-dc0c-2f4943157cba"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "tensor(1., requires_grad=True)"
            ]
          },
          "metadata": {},
          "execution_count": 160
        }
      ],
      "source": [
        "w"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 161,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ZwUFjr4jCkGs",
        "outputId": "50ce540c-61c3-4155-8315-4164470ffd3d"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "tensor(0., requires_grad=True)"
            ]
          },
          "metadata": {},
          "execution_count": 161
        }
      ],
      "source": [
        "b"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 162,
      "metadata": {
        "id": "E5qhTSzKCoIO"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import torch\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.preprocessing import LabelEncoder"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 163,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 273
        },
        "id": "f8gtLIcwiZyI",
        "outputId": "74990ec2-684b-49cd-be2a-0bbd755ed88a"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "         id diagnosis  radius_mean  texture_mean  perimeter_mean  area_mean  \\\n",
              "0    842302         M        17.99         10.38          122.80     1001.0   \n",
              "1    842517         M        20.57         17.77          132.90     1326.0   \n",
              "2  84300903         M        19.69         21.25          130.00     1203.0   \n",
              "3  84348301         M        11.42         20.38           77.58      386.1   \n",
              "4  84358402         M        20.29         14.34          135.10     1297.0   \n",
              "\n",
              "   smoothness_mean  compactness_mean  concavity_mean  concave points_mean  \\\n",
              "0          0.11840           0.27760          0.3001              0.14710   \n",
              "1          0.08474           0.07864          0.0869              0.07017   \n",
              "2          0.10960           0.15990          0.1974              0.12790   \n",
              "3          0.14250           0.28390          0.2414              0.10520   \n",
              "4          0.10030           0.13280          0.1980              0.10430   \n",
              "\n",
              "   ...  texture_worst  perimeter_worst  area_worst  smoothness_worst  \\\n",
              "0  ...          17.33           184.60      2019.0            0.1622   \n",
              "1  ...          23.41           158.80      1956.0            0.1238   \n",
              "2  ...          25.53           152.50      1709.0            0.1444   \n",
              "3  ...          26.50            98.87       567.7            0.2098   \n",
              "4  ...          16.67           152.20      1575.0            0.1374   \n",
              "\n",
              "   compactness_worst  concavity_worst  concave points_worst  symmetry_worst  \\\n",
              "0             0.6656           0.7119                0.2654          0.4601   \n",
              "1             0.1866           0.2416                0.1860          0.2750   \n",
              "2             0.4245           0.4504                0.2430          0.3613   \n",
              "3             0.8663           0.6869                0.2575          0.6638   \n",
              "4             0.2050           0.4000                0.1625          0.2364   \n",
              "\n",
              "   fractal_dimension_worst  Unnamed: 32  \n",
              "0                  0.11890          NaN  \n",
              "1                  0.08902          NaN  \n",
              "2                  0.08758          NaN  \n",
              "3                  0.17300          NaN  \n",
              "4                  0.07678          NaN  \n",
              "\n",
              "[5 rows x 33 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-b7833dd1-938b-489b-860f-a6b1fa250b40\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>id</th>\n",
              "      <th>diagnosis</th>\n",
              "      <th>radius_mean</th>\n",
              "      <th>texture_mean</th>\n",
              "      <th>perimeter_mean</th>\n",
              "      <th>area_mean</th>\n",
              "      <th>smoothness_mean</th>\n",
              "      <th>compactness_mean</th>\n",
              "      <th>concavity_mean</th>\n",
              "      <th>concave points_mean</th>\n",
              "      <th>...</th>\n",
              "      <th>texture_worst</th>\n",
              "      <th>perimeter_worst</th>\n",
              "      <th>area_worst</th>\n",
              "      <th>smoothness_worst</th>\n",
              "      <th>compactness_worst</th>\n",
              "      <th>concavity_worst</th>\n",
              "      <th>concave points_worst</th>\n",
              "      <th>symmetry_worst</th>\n",
              "      <th>fractal_dimension_worst</th>\n",
              "      <th>Unnamed: 32</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>842302</td>\n",
              "      <td>M</td>\n",
              "      <td>17.99</td>\n",
              "      <td>10.38</td>\n",
              "      <td>122.80</td>\n",
              "      <td>1001.0</td>\n",
              "      <td>0.11840</td>\n",
              "      <td>0.27760</td>\n",
              "      <td>0.3001</td>\n",
              "      <td>0.14710</td>\n",
              "      <td>...</td>\n",
              "      <td>17.33</td>\n",
              "      <td>184.60</td>\n",
              "      <td>2019.0</td>\n",
              "      <td>0.1622</td>\n",
              "      <td>0.6656</td>\n",
              "      <td>0.7119</td>\n",
              "      <td>0.2654</td>\n",
              "      <td>0.4601</td>\n",
              "      <td>0.11890</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>842517</td>\n",
              "      <td>M</td>\n",
              "      <td>20.57</td>\n",
              "      <td>17.77</td>\n",
              "      <td>132.90</td>\n",
              "      <td>1326.0</td>\n",
              "      <td>0.08474</td>\n",
              "      <td>0.07864</td>\n",
              "      <td>0.0869</td>\n",
              "      <td>0.07017</td>\n",
              "      <td>...</td>\n",
              "      <td>23.41</td>\n",
              "      <td>158.80</td>\n",
              "      <td>1956.0</td>\n",
              "      <td>0.1238</td>\n",
              "      <td>0.1866</td>\n",
              "      <td>0.2416</td>\n",
              "      <td>0.1860</td>\n",
              "      <td>0.2750</td>\n",
              "      <td>0.08902</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>84300903</td>\n",
              "      <td>M</td>\n",
              "      <td>19.69</td>\n",
              "      <td>21.25</td>\n",
              "      <td>130.00</td>\n",
              "      <td>1203.0</td>\n",
              "      <td>0.10960</td>\n",
              "      <td>0.15990</td>\n",
              "      <td>0.1974</td>\n",
              "      <td>0.12790</td>\n",
              "      <td>...</td>\n",
              "      <td>25.53</td>\n",
              "      <td>152.50</td>\n",
              "      <td>1709.0</td>\n",
              "      <td>0.1444</td>\n",
              "      <td>0.4245</td>\n",
              "      <td>0.4504</td>\n",
              "      <td>0.2430</td>\n",
              "      <td>0.3613</td>\n",
              "      <td>0.08758</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>84348301</td>\n",
              "      <td>M</td>\n",
              "      <td>11.42</td>\n",
              "      <td>20.38</td>\n",
              "      <td>77.58</td>\n",
              "      <td>386.1</td>\n",
              "      <td>0.14250</td>\n",
              "      <td>0.28390</td>\n",
              "      <td>0.2414</td>\n",
              "      <td>0.10520</td>\n",
              "      <td>...</td>\n",
              "      <td>26.50</td>\n",
              "      <td>98.87</td>\n",
              "      <td>567.7</td>\n",
              "      <td>0.2098</td>\n",
              "      <td>0.8663</td>\n",
              "      <td>0.6869</td>\n",
              "      <td>0.2575</td>\n",
              "      <td>0.6638</td>\n",
              "      <td>0.17300</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>84358402</td>\n",
              "      <td>M</td>\n",
              "      <td>20.29</td>\n",
              "      <td>14.34</td>\n",
              "      <td>135.10</td>\n",
              "      <td>1297.0</td>\n",
              "      <td>0.10030</td>\n",
              "      <td>0.13280</td>\n",
              "      <td>0.1980</td>\n",
              "      <td>0.10430</td>\n",
              "      <td>...</td>\n",
              "      <td>16.67</td>\n",
              "      <td>152.20</td>\n",
              "      <td>1575.0</td>\n",
              "      <td>0.1374</td>\n",
              "      <td>0.2050</td>\n",
              "      <td>0.4000</td>\n",
              "      <td>0.1625</td>\n",
              "      <td>0.2364</td>\n",
              "      <td>0.07678</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 33 columns</p>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-b7833dd1-938b-489b-860f-a6b1fa250b40')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-b7833dd1-938b-489b-860f-a6b1fa250b40 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-b7833dd1-938b-489b-860f-a6b1fa250b40');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    <div id=\"df-e79b7378-1e1a-49fb-b66a-5b0d190aa543\">\n",
              "      <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-e79b7378-1e1a-49fb-b66a-5b0d190aa543')\"\n",
              "                title=\"Suggest charts\"\n",
              "                style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "      </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "      <script>\n",
              "        async function quickchart(key) {\n",
              "          const quickchartButtonEl =\n",
              "            document.querySelector('#' + key + ' button');\n",
              "          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "          quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "          try {\n",
              "            const charts = await google.colab.kernel.invokeFunction(\n",
              "                'suggestCharts', [key], {});\n",
              "          } catch (error) {\n",
              "            console.error('Error during call to suggestCharts:', error);\n",
              "          }\n",
              "          quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "          quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "        }\n",
              "        (() => {\n",
              "          let quickchartButtonEl =\n",
              "            document.querySelector('#df-e79b7378-1e1a-49fb-b66a-5b0d190aa543 button');\n",
              "          quickchartButtonEl.style.display =\n",
              "            google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "        })();\n",
              "      </script>\n",
              "    </div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "df"
            }
          },
          "metadata": {},
          "execution_count": 163
        }
      ],
      "source": [
        "df = pd.read_csv('https://raw.githubusercontent.com/gscdit/Breast-cancer-Detection/refs/heads/master/data.csv')\n",
        "df.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 164,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zoDZQRa0mNU2",
        "outputId": "a67aa249-a805-4dfd-949d-cad55c2e57c0"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(569, 33)"
            ]
          },
          "metadata": {},
          "execution_count": 164
        }
      ],
      "source": [
        "df.shape\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 165,
      "metadata": {
        "id": "CJngGD9CmRRK"
      },
      "outputs": [],
      "source": [
        "df.drop(columns=['id', 'Unnamed: 32'], inplace=True, errors='ignore')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 166,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 273
        },
        "id": "o8Jux_dVmpnP",
        "outputId": "fd6762f4-c957-47f4-c47e-3ae4a82b38a6"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "  diagnosis  radius_mean  texture_mean  perimeter_mean  area_mean  \\\n",
              "0         M        17.99         10.38          122.80     1001.0   \n",
              "1         M        20.57         17.77          132.90     1326.0   \n",
              "2         M        19.69         21.25          130.00     1203.0   \n",
              "3         M        11.42         20.38           77.58      386.1   \n",
              "4         M        20.29         14.34          135.10     1297.0   \n",
              "\n",
              "   smoothness_mean  compactness_mean  concavity_mean  concave points_mean  \\\n",
              "0          0.11840           0.27760          0.3001              0.14710   \n",
              "1          0.08474           0.07864          0.0869              0.07017   \n",
              "2          0.10960           0.15990          0.1974              0.12790   \n",
              "3          0.14250           0.28390          0.2414              0.10520   \n",
              "4          0.10030           0.13280          0.1980              0.10430   \n",
              "\n",
              "   symmetry_mean  ...  radius_worst  texture_worst  perimeter_worst  \\\n",
              "0         0.2419  ...         25.38          17.33           184.60   \n",
              "1         0.1812  ...         24.99          23.41           158.80   \n",
              "2         0.2069  ...         23.57          25.53           152.50   \n",
              "3         0.2597  ...         14.91          26.50            98.87   \n",
              "4         0.1809  ...         22.54          16.67           152.20   \n",
              "\n",
              "   area_worst  smoothness_worst  compactness_worst  concavity_worst  \\\n",
              "0      2019.0            0.1622             0.6656           0.7119   \n",
              "1      1956.0            0.1238             0.1866           0.2416   \n",
              "2      1709.0            0.1444             0.4245           0.4504   \n",
              "3       567.7            0.2098             0.8663           0.6869   \n",
              "4      1575.0            0.1374             0.2050           0.4000   \n",
              "\n",
              "   concave points_worst  symmetry_worst  fractal_dimension_worst  \n",
              "0                0.2654          0.4601                  0.11890  \n",
              "1                0.1860          0.2750                  0.08902  \n",
              "2                0.2430          0.3613                  0.08758  \n",
              "3                0.2575          0.6638                  0.17300  \n",
              "4                0.1625          0.2364                  0.07678  \n",
              "\n",
              "[5 rows x 31 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-aee2d015-5a43-4799-810e-42a315948a33\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>diagnosis</th>\n",
              "      <th>radius_mean</th>\n",
              "      <th>texture_mean</th>\n",
              "      <th>perimeter_mean</th>\n",
              "      <th>area_mean</th>\n",
              "      <th>smoothness_mean</th>\n",
              "      <th>compactness_mean</th>\n",
              "      <th>concavity_mean</th>\n",
              "      <th>concave points_mean</th>\n",
              "      <th>symmetry_mean</th>\n",
              "      <th>...</th>\n",
              "      <th>radius_worst</th>\n",
              "      <th>texture_worst</th>\n",
              "      <th>perimeter_worst</th>\n",
              "      <th>area_worst</th>\n",
              "      <th>smoothness_worst</th>\n",
              "      <th>compactness_worst</th>\n",
              "      <th>concavity_worst</th>\n",
              "      <th>concave points_worst</th>\n",
              "      <th>symmetry_worst</th>\n",
              "      <th>fractal_dimension_worst</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>M</td>\n",
              "      <td>17.99</td>\n",
              "      <td>10.38</td>\n",
              "      <td>122.80</td>\n",
              "      <td>1001.0</td>\n",
              "      <td>0.11840</td>\n",
              "      <td>0.27760</td>\n",
              "      <td>0.3001</td>\n",
              "      <td>0.14710</td>\n",
              "      <td>0.2419</td>\n",
              "      <td>...</td>\n",
              "      <td>25.38</td>\n",
              "      <td>17.33</td>\n",
              "      <td>184.60</td>\n",
              "      <td>2019.0</td>\n",
              "      <td>0.1622</td>\n",
              "      <td>0.6656</td>\n",
              "      <td>0.7119</td>\n",
              "      <td>0.2654</td>\n",
              "      <td>0.4601</td>\n",
              "      <td>0.11890</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>M</td>\n",
              "      <td>20.57</td>\n",
              "      <td>17.77</td>\n",
              "      <td>132.90</td>\n",
              "      <td>1326.0</td>\n",
              "      <td>0.08474</td>\n",
              "      <td>0.07864</td>\n",
              "      <td>0.0869</td>\n",
              "      <td>0.07017</td>\n",
              "      <td>0.1812</td>\n",
              "      <td>...</td>\n",
              "      <td>24.99</td>\n",
              "      <td>23.41</td>\n",
              "      <td>158.80</td>\n",
              "      <td>1956.0</td>\n",
              "      <td>0.1238</td>\n",
              "      <td>0.1866</td>\n",
              "      <td>0.2416</td>\n",
              "      <td>0.1860</td>\n",
              "      <td>0.2750</td>\n",
              "      <td>0.08902</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>M</td>\n",
              "      <td>19.69</td>\n",
              "      <td>21.25</td>\n",
              "      <td>130.00</td>\n",
              "      <td>1203.0</td>\n",
              "      <td>0.10960</td>\n",
              "      <td>0.15990</td>\n",
              "      <td>0.1974</td>\n",
              "      <td>0.12790</td>\n",
              "      <td>0.2069</td>\n",
              "      <td>...</td>\n",
              "      <td>23.57</td>\n",
              "      <td>25.53</td>\n",
              "      <td>152.50</td>\n",
              "      <td>1709.0</td>\n",
              "      <td>0.1444</td>\n",
              "      <td>0.4245</td>\n",
              "      <td>0.4504</td>\n",
              "      <td>0.2430</td>\n",
              "      <td>0.3613</td>\n",
              "      <td>0.08758</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>M</td>\n",
              "      <td>11.42</td>\n",
              "      <td>20.38</td>\n",
              "      <td>77.58</td>\n",
              "      <td>386.1</td>\n",
              "      <td>0.14250</td>\n",
              "      <td>0.28390</td>\n",
              "      <td>0.2414</td>\n",
              "      <td>0.10520</td>\n",
              "      <td>0.2597</td>\n",
              "      <td>...</td>\n",
              "      <td>14.91</td>\n",
              "      <td>26.50</td>\n",
              "      <td>98.87</td>\n",
              "      <td>567.7</td>\n",
              "      <td>0.2098</td>\n",
              "      <td>0.8663</td>\n",
              "      <td>0.6869</td>\n",
              "      <td>0.2575</td>\n",
              "      <td>0.6638</td>\n",
              "      <td>0.17300</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>M</td>\n",
              "      <td>20.29</td>\n",
              "      <td>14.34</td>\n",
              "      <td>135.10</td>\n",
              "      <td>1297.0</td>\n",
              "      <td>0.10030</td>\n",
              "      <td>0.13280</td>\n",
              "      <td>0.1980</td>\n",
              "      <td>0.10430</td>\n",
              "      <td>0.1809</td>\n",
              "      <td>...</td>\n",
              "      <td>22.54</td>\n",
              "      <td>16.67</td>\n",
              "      <td>152.20</td>\n",
              "      <td>1575.0</td>\n",
              "      <td>0.1374</td>\n",
              "      <td>0.2050</td>\n",
              "      <td>0.4000</td>\n",
              "      <td>0.1625</td>\n",
              "      <td>0.2364</td>\n",
              "      <td>0.07678</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 31 columns</p>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-aee2d015-5a43-4799-810e-42a315948a33')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-aee2d015-5a43-4799-810e-42a315948a33 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-aee2d015-5a43-4799-810e-42a315948a33');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    <div id=\"df-04c6ea81-425b-4915-952c-4230d0afd83c\">\n",
              "      <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-04c6ea81-425b-4915-952c-4230d0afd83c')\"\n",
              "                title=\"Suggest charts\"\n",
              "                style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "      </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "      <script>\n",
              "        async function quickchart(key) {\n",
              "          const quickchartButtonEl =\n",
              "            document.querySelector('#' + key + ' button');\n",
              "          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "          quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "          try {\n",
              "            const charts = await google.colab.kernel.invokeFunction(\n",
              "                'suggestCharts', [key], {});\n",
              "          } catch (error) {\n",
              "            console.error('Error during call to suggestCharts:', error);\n",
              "          }\n",
              "          quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "          quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "        }\n",
              "        (() => {\n",
              "          let quickchartButtonEl =\n",
              "            document.querySelector('#df-04c6ea81-425b-4915-952c-4230d0afd83c button');\n",
              "          quickchartButtonEl.style.display =\n",
              "            google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "        })();\n",
              "      </script>\n",
              "    </div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "df"
            }
          },
          "metadata": {},
          "execution_count": 166
        }
      ],
      "source": [
        "df.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 167,
      "metadata": {
        "id": "Wk0fUfN0nICE"
      },
      "outputs": [],
      "source": [
        "x_train, x_test, y_train,y_test=train_test_split(df.iloc[:,1:], df.iloc[:,0],test_size=0.2)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 168,
      "metadata": {
        "id": "hLXkRAUrn8mp"
      },
      "outputs": [],
      "source": [
        "scalar = StandardScaler()\n",
        "x_train = scalar.fit_transform(x_train)\n",
        "x_test = scalar.transform(x_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 169,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "67H0X612o7gG",
        "outputId": "f3202ced-b9a5-4af5-88f8-45a27a58fb29"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[-0.16196319, -0.93823877, -0.17996517, ..., -0.16423974,\n",
              "         0.79823995,  0.70218189],\n",
              "       [-0.28965611, -0.77954391, -0.349045  , ..., -1.36381078,\n",
              "        -0.381509  ,  0.00606113],\n",
              "       [ 1.54523564,  0.7197672 ,  1.4999637 , ...,  1.8260933 ,\n",
              "        -0.19396738, -0.42172257],\n",
              "       ...,\n",
              "       [ 0.19613218, -0.36504242,  0.16544076, ...,  0.75035857,\n",
              "         0.84151879, -0.68172617],\n",
              "       [ 1.90333101,  0.33368867,  1.82604623, ...,  1.66495689,\n",
              "        -0.99542727, -0.52727959],\n",
              "       [-0.83096305, -1.02587622, -0.85990763, ..., -1.4880947 ,\n",
              "        -0.11863015, -0.53116853]])"
            ]
          },
          "metadata": {},
          "execution_count": 169
        }
      ],
      "source": [
        "x_train"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 170,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 458
        },
        "id": "cDcmiRw9pGr2",
        "outputId": "fe056f3c-c71d-4812-8992-ca2e31135f82"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "378    B\n",
              "306    B\n",
              "18     M\n",
              "555    B\n",
              "236    M\n",
              "      ..\n",
              "239    M\n",
              "150    B\n",
              "138    M\n",
              "449    M\n",
              "333    B\n",
              "Name: diagnosis, Length: 455, dtype: object"
            ],
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>diagnosis</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>378</th>\n",
              "      <td>B</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>306</th>\n",
              "      <td>B</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>18</th>\n",
              "      <td>M</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>555</th>\n",
              "      <td>B</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>236</th>\n",
              "      <td>M</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>239</th>\n",
              "      <td>M</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>150</th>\n",
              "      <td>B</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>138</th>\n",
              "      <td>M</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>449</th>\n",
              "      <td>M</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>333</th>\n",
              "      <td>B</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>455 rows × 1 columns</p>\n",
              "</div><br><label><b>dtype:</b> object</label>"
            ]
          },
          "metadata": {},
          "execution_count": 170
        }
      ],
      "source": [
        "y_train"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 171,
      "metadata": {
        "id": "zzCH4k3YpLYt"
      },
      "outputs": [],
      "source": [
        "encoder = LabelEncoder()\n",
        "y_train = encoder.fit_transform(y_train)\n",
        "y_test = encoder.transform(y_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 172,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bFTo6LO9phKe",
        "outputId": "4947ac75-eec6-465d-9321-589b014715d8"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0,\n",
              "       0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,\n",
              "       0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,\n",
              "       1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,\n",
              "       0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1,\n",
              "       0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1,\n",
              "       0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,\n",
              "       0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0,\n",
              "       1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0,\n",
              "       0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1,\n",
              "       0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,\n",
              "       0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0,\n",
              "       1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0,\n",
              "       0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,\n",
              "       1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0,\n",
              "       0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,\n",
              "       0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0,\n",
              "       0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0,\n",
              "       0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1,\n",
              "       1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,\n",
              "       0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0])"
            ]
          },
          "metadata": {},
          "execution_count": 172
        }
      ],
      "source": [
        "y_train"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 173,
      "metadata": {
        "id": "1pwBgLnapnTH"
      },
      "outputs": [],
      "source": [
        "x_train_tensor = torch.from_numpy(x_train)\n",
        "x_test_tensor = torch.from_numpy(x_test)\n",
        "y_train_tensor = torch.from_numpy(y_train)\n",
        "y_test_tensor = torch.from_numpy(y_test)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 174,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CuUCyBPiq3Dc",
        "outputId": "ea4bef87-0b9f-41b6-a218-39a4d67b357e"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "tensor([[-0.1620, -0.9382, -0.1800,  ..., -0.1642,  0.7982,  0.7022],\n",
              "        [-0.2897, -0.7795, -0.3490,  ..., -1.3638, -0.3815,  0.0061],\n",
              "        [ 1.5452,  0.7198,  1.5000,  ...,  1.8261, -0.1940, -0.4217],\n",
              "        ...,\n",
              "        [ 0.1961, -0.3650,  0.1654,  ...,  0.7504,  0.8415, -0.6817],\n",
              "        [ 1.9033,  0.3337,  1.8260,  ...,  1.6650, -0.9954, -0.5273],\n",
              "        [-0.8310, -1.0259, -0.8599,  ..., -1.4881, -0.1186, -0.5312]],\n",
              "       dtype=torch.float64)"
            ]
          },
          "metadata": {},
          "execution_count": 174
        }
      ],
      "source": [
        "x_train_tensor"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 175,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "H9kekMrhrFgb",
        "outputId": "caad3693-b181-4aad-d82f-f44965c4e757"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "torch.Size([455])"
            ]
          },
          "metadata": {},
          "execution_count": 175
        }
      ],
      "source": [
        "y_train_tensor.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 176,
      "metadata": {
        "id": "mFMN19SxrjBh"
      },
      "outputs": [],
      "source": [
        "class MySimpleNN():\n",
        "  def __init__(self, x):\n",
        "    self.weights  = torch.rand(x.shape [1], 1, dtype=torch.float64, requires_grad=True)\n",
        "    self.bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)\n",
        "\n",
        "  def forward (self, X ):\n",
        "    z = torch.matmul(X, self.weights) + self.bias\n",
        "    y_pred = torch.sigmoid(z)\n",
        "    return y_pred\n",
        "\n",
        "  def loss_function(self, y_pred, y):\n",
        "    #clamp predictions to avoid log(0)\n",
        "    epsilon = 1e-7\n",
        "    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)\n",
        "\n",
        "    # Calculate loss\n",
        "    loss = -(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred)).mean()\n",
        "    return loss"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 177,
      "metadata": {
        "id": "FHJ_CHizsRhe"
      },
      "outputs": [],
      "source": [
        "learning_rate = 0.1\n",
        "epochs = 25"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 178,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "collapsed": true,
        "id": "fUttdznitxK2",
        "outputId": "fa708bc2-69f2-4379-96f9-603b20b374ef"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch: 1, Loss: 3.511938517506579\n",
            "Epoch: 2, Loss: 3.3786757850285296\n",
            "Epoch: 3, Loss: 3.242555272389433\n",
            "Epoch: 4, Loss: 3.1010966526501216\n",
            "Epoch: 5, Loss: 2.955676244000585\n",
            "Epoch: 6, Loss: 2.805077275583171\n",
            "Epoch: 7, Loss: 2.650252205524109\n",
            "Epoch: 8, Loss: 2.4932351452518158\n",
            "Epoch: 9, Loss: 2.3325642531764013\n",
            "Epoch: 10, Loss: 2.1733898042036937\n",
            "Epoch: 11, Loss: 2.019912655143022\n",
            "Epoch: 12, Loss: 1.8679137097289675\n",
            "Epoch: 13, Loss: 1.7161894093995842\n",
            "Epoch: 14, Loss: 1.5737442787021807\n",
            "Epoch: 15, Loss: 1.4403845149189034\n",
            "Epoch: 16, Loss: 1.319706736639421\n",
            "Epoch: 17, Loss: 1.2097865258839589\n",
            "Epoch: 18, Loss: 1.1151916488206328\n",
            "Epoch: 19, Loss: 1.0362007495505094\n",
            "Epoch: 20, Loss: 0.9723840243065514\n",
            "Epoch: 21, Loss: 0.9224662447535913\n",
            "Epoch: 22, Loss: 0.8844059984106829\n",
            "Epoch: 23, Loss: 0.8557388946923008\n",
            "Epoch: 24, Loss: 0.8340353028264561\n",
            "Epoch: 25, Loss: 0.8172527017625182\n"
          ]
        }
      ],
      "source": [
        "# create model\n",
        "model = MySimpleNN(x_train_tensor)\n",
        "\n",
        "# define loop\n",
        "for epoch in range(epochs):\n",
        "    # forward pass\n",
        "    y_pred = model.forward(x_train_tensor)\n",
        "\n",
        "    # loss calculate\n",
        "    loss = model.loss_function(y_pred, y_train_tensor)\n",
        "\n",
        "\n",
        "    # backward pass\n",
        "    loss.backward()\n",
        "\n",
        "    # parameters update\n",
        "    with torch.no_grad():\n",
        "        model.weights -= learning_rate * model.weights.grad\n",
        "        model.bias -= learning_rate * model.bias.grad\n",
        "\n",
        "        # zero gradients\n",
        "        model.weights.grad.zero_()\n",
        "        model.bias.grad.zero_()\n",
        "\n",
        "    # print loss in each epoch\n",
        "    print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 179,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3lQfZCddujgO",
        "outputId": "1837dbe0-3dab-4a17-d228-e5fd9847456b"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "tensor([-0.1140], dtype=torch.float64, requires_grad=True)"
            ]
          },
          "metadata": {},
          "execution_count": 179
        }
      ],
      "source": [
        "model.bias"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 180,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sfhAvVw9eXN9",
        "outputId": "e3e41187-3a77-4dbd-91ec-343574f81253"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: 0.5355494022369385\n"
          ]
        }
      ],
      "source": [
        "# model evaluation\n",
        "with torch.no_grad():\n",
        "    y_pred = model.forward(x_test_tensor)\n",
        "    y_pred = (y_pred > 0.5).float()\n",
        "    accuracy = (y_pred == y_test_tensor).float().mean()\n",
        "    print(f'Accuracy: {accuracy.item()}')\n",
        "\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 181,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qQB1DMGagxth",
        "outputId": "b6cd769a-bce3-421f-bbd7-68ce539894fc"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch: 0, Loss: 1.0037\n",
            "Epoch: 1, Loss: 0.9923\n",
            "Epoch: 2, Loss: 0.9816\n",
            "Epoch: 3, Loss: 0.9715\n",
            "Epoch: 4, Loss: 0.9619\n"
          ]
        }
      ],
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "\n",
        "# 1. DEFINE DATA (This is what was missing)\n",
        "# Creating dummy data for demonstration\n",
        "x_train_tensor = torch.randn(10, 5)  # 10 samples, 5 features\n",
        "y_train_tensor = torch.randn(10, 1)  # 10 target values\n",
        "\n",
        "# 2. DEFINE MODEL\n",
        "class MySimpleNN(nn.Module):\n",
        "    def __init__(self):\n",
        "        super().__init__()\n",
        "        self.linear = nn.Linear(5, 1)\n",
        "        self.loss_function = nn.MSELoss()\n",
        "\n",
        "    def forward(self, x):\n",
        "        return self.linear(x)\n",
        "\n",
        "# Initialize model and optimizer\n",
        "model = MySimpleNN()\n",
        "optimizer = optim.SGD(model.parameters(), lr=0.01)\n",
        "\n",
        "# 3. TRAINING LOOP\n",
        "epochs = 5\n",
        "for epoch in range(epochs):\n",
        "    # Forward pass\n",
        "    y_pred = model(x_train_tensor) # Note: calling model() is better than model.forward()\n",
        "\n",
        "    # Loss calculate\n",
        "    loss = model.loss_function(y_pred, y_train_tensor)\n",
        "\n",
        "    # Backward pass\n",
        "    optimizer.zero_grad() # Clear old gradients\n",
        "    loss.backward()       # Compute new gradients\n",
        "\n",
        "    # Parameters update\n",
        "    optimizer.step()      # Update weights\n",
        "\n",
        "    print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 182,
      "metadata": {
        "id": "MBERrNtKAaMo"
      },
      "outputs": [],
      "source": [
        "from sklearn.datasets import make_classification\n",
        "import torch\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 183,
      "metadata": {
        "id": "ZXQIzu2ZGR_F"
      },
      "outputs": [],
      "source": [
        "# step 1 : create a synthetic classification dataset using sklearn\n",
        "X,Y = make_classification(\n",
        "    n_samples=10,       # Number of samples\n",
        "    n_features=2,        # Number of features\n",
        "    n_informative=2,    # Number of informative features\n",
        "    n_redundant=0,      # Number of redundant features\n",
        "    n_classes=2,        # Number of  classes\n",
        "    random_state=42     # For reproducibility\n",
        ")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 184,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "D495OiCUIBKe",
        "outputId": "f6cf10a0-e431-45ab-e360-3a973fe5e9a9"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[ 1.06833894, -0.97007347],\n",
              "       [-1.14021544, -0.83879234],\n",
              "       [-2.8953973 ,  1.97686236],\n",
              "       [-0.72063436, -0.96059253],\n",
              "       [-1.96287438, -0.99225135],\n",
              "       [-0.9382051 , -0.54304815],\n",
              "       [ 1.72725924, -1.18582677],\n",
              "       [ 1.77736657,  1.51157598],\n",
              "       [ 1.89969252,  0.83444483],\n",
              "       [-0.58723065, -1.97171753]])"
            ]
          },
          "metadata": {},
          "execution_count": 184
        }
      ],
      "source": [
        "X"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 185,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8K9ns4bWIjGJ",
        "outputId": "e4804489-23b1-4bcd-9407-107b263afe60"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(10, 2)"
            ]
          },
          "metadata": {},
          "execution_count": 185
        }
      ],
      "source": [
        "X.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 186,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YogdG-xzIni8",
        "outputId": "ea50b7b8-24ba-419e-a0a5-985ce20616fc"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([1, 0, 0, 0, 0, 1, 1, 1, 1, 0])"
            ]
          },
          "metadata": {},
          "execution_count": 186
        }
      ],
      "source": [
        "Y"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 187,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MeQ9vS-gIpFX",
        "outputId": "da913043-ed8d-457b-a186-6df9c6748b60"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(10,)"
            ]
          },
          "metadata": {},
          "execution_count": 187
        }
      ],
      "source": [
        "Y.shape"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 188,
      "metadata": {
        "id": "-ZO3sE7NIycl"
      },
      "outputs": [],
      "source": [
        "# convert the data to pytorch tensors\n",
        "X = torch.tensor(X, dtype=torch.float32)\n",
        "Y = torch.tensor(Y, dtype=torch.long)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 189,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MJzhreREJMzq",
        "outputId": "896c3d7e-bed7-4dda-8698-75c87094a2d2"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "tensor([[ 1.0683, -0.9701],\n",
              "        [-1.1402, -0.8388],\n",
              "        [-2.8954,  1.9769],\n",
              "        [-0.7206, -0.9606],\n",
              "        [-1.9629, -0.9923],\n",
              "        [-0.9382, -0.5430],\n",
              "        [ 1.7273, -1.1858],\n",
              "        [ 1.7774,  1.5116],\n",
              "        [ 1.8997,  0.8344],\n",
              "        [-0.5872, -1.9717]])"
            ]
          },
          "metadata": {},
          "execution_count": 189
        }
      ],
      "source": [
        "X"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 190,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bLEU4PiMJQ0X",
        "outputId": "b3425d11-5606-4783-a9c1-c9113661cf70"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "tensor([1, 0, 0, 0, 0, 1, 1, 1, 1, 0])"
            ]
          },
          "metadata": {},
          "execution_count": 190
        }
      ],
      "source": [
        "Y"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 191,
      "metadata": {
        "id": "K_tm10E1JUex"
      },
      "outputs": [],
      "source": [
        "from torch.utils.data import Dataset, DataLoader"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 192,
      "metadata": {
        "id": "Vh0kpG71JhZV"
      },
      "outputs": [],
      "source": [
        "class CustomDataset(Dataset):\n",
        "\n",
        "  def __init__(self, features, labels):\n",
        "\n",
        "\n",
        "    self.features = features\n",
        "    self.labels = labels\n",
        "\n",
        "  def __len__(self):\n",
        "\n",
        "    return self.features.shape[0]\n",
        "\n",
        "  def __getitem__(self, index):\n",
        "\n",
        "    return self.features[index], self.labels[index]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 193,
      "metadata": {
        "id": "hsJUIxI2KlcL"
      },
      "outputs": [],
      "source": [
        "dataset = CustomDataset (X, Y)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 194,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "UcFZWYMeLMw1",
        "outputId": "572e8040-374f-4caa-b485-d3825434ac14"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "10"
            ]
          },
          "metadata": {},
          "execution_count": 194
        }
      ],
      "source": [
        "len(dataset)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 195,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "onW5tDetLduI",
        "outputId": "785fc48a-024a-439a-83bc-2c48cc28f4eb"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(tensor([-2.8954,  1.9769]), tensor(0))"
            ]
          },
          "metadata": {},
          "execution_count": 195
        }
      ],
      "source": [
        "dataset[2]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 196,
      "metadata": {
        "id": "kRh4wwxFLvMM"
      },
      "outputs": [],
      "source": [
        "dataloader = DataLoader(dataset, batch_size=2, shuffle=False)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 197,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VnoKiwMeMFqv",
        "outputId": "87dfc78f-82f7-42c4-f6e6-470c6524ee9f"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "tensor([[ 1.0683, -0.9701],\n",
            "        [-1.1402, -0.8388]])\n",
            "tensor([1, 0])\n",
            "--------------------------------------------------\n",
            "tensor([[-2.8954,  1.9769],\n",
            "        [-0.7206, -0.9606]])\n",
            "tensor([0, 0])\n",
            "--------------------------------------------------\n",
            "tensor([[-1.9629, -0.9923],\n",
            "        [-0.9382, -0.5430]])\n",
            "tensor([0, 1])\n",
            "--------------------------------------------------\n",
            "tensor([[ 1.7273, -1.1858],\n",
            "        [ 1.7774,  1.5116]])\n",
            "tensor([1, 1])\n",
            "--------------------------------------------------\n",
            "tensor([[ 1.8997,  0.8344],\n",
            "        [-0.5872, -1.9717]])\n",
            "tensor([1, 0])\n",
            "--------------------------------------------------\n"
          ]
        }
      ],
      "source": [
        "for batch_features, batch_labels in dataloader :\n",
        "\n",
        "  print(batch_features)\n",
        "  print(batch_labels)\n",
        "  print(\"-\"*50)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 198,
      "metadata": {
        "id": "8VxylSjTrZF0"
      },
      "outputs": [],
      "source": [
        "import matplotlib.pyplot as plt\n",
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "import torch\n",
        "from torch.utils.data import Dataset, DataLoader\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 199,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tc7FPj7T3IxB",
        "outputId": "198d512a-7eba-4a94-b1a6-1520c381bfcd"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<torch._C.Generator at 0x7a87f01d5910>"
            ]
          },
          "metadata": {},
          "execution_count": 199
        }
      ],
      "source": [
        "# set random seed for reproducibility\n",
        "torch.manual_seed(42)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 200,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 235
        },
        "id": "y9zUlWX1ssgt",
        "outputId": "01cf79b3-162b-49e2-ac71-8e4a645dfede"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   7  0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  ...  0.658  0.659  0.660  \\\n",
              "0  2  0    0    0    0    0    0    0    0    0  ...      0      0      0   \n",
              "1  1  0    0    0    0    0    0    0    0    0  ...      0      0      0   \n",
              "2  0  0    0    0    0    0    0    0    0    0  ...      0      0      0   \n",
              "3  4  0    0    0    0    0    0    0    0    0  ...      0      0      0   \n",
              "4  1  0    0    0    0    0    0    0    0    0  ...      0      0      0   \n",
              "\n",
              "   0.661  0.662  0.663  0.664  0.665  0.666  0.667  \n",
              "0      0      0      0      0      0      0      0  \n",
              "1      0      0      0      0      0      0      0  \n",
              "2      0      0      0      0      0      0      0  \n",
              "3      0      0      0      0      0      0      0  \n",
              "4      0      0      0      0      0      0      0  \n",
              "\n",
              "[5 rows x 785 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-b76b6099-4ef0-4af0-b7db-f1cd94fe7dfe\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>7</th>\n",
              "      <th>0</th>\n",
              "      <th>0.1</th>\n",
              "      <th>0.2</th>\n",
              "      <th>0.3</th>\n",
              "      <th>0.4</th>\n",
              "      <th>0.5</th>\n",
              "      <th>0.6</th>\n",
              "      <th>0.7</th>\n",
              "      <th>0.8</th>\n",
              "      <th>...</th>\n",
              "      <th>0.658</th>\n",
              "      <th>0.659</th>\n",
              "      <th>0.660</th>\n",
              "      <th>0.661</th>\n",
              "      <th>0.662</th>\n",
              "      <th>0.663</th>\n",
              "      <th>0.664</th>\n",
              "      <th>0.665</th>\n",
              "      <th>0.666</th>\n",
              "      <th>0.667</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>4</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 785 columns</p>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-b76b6099-4ef0-4af0-b7db-f1cd94fe7dfe')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-b76b6099-4ef0-4af0-b7db-f1cd94fe7dfe button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-b76b6099-4ef0-4af0-b7db-f1cd94fe7dfe');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    <div id=\"df-24c01f7d-a426-4b65-b97a-ca94a92124cb\">\n",
              "      <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-24c01f7d-a426-4b65-b97a-ca94a92124cb')\"\n",
              "                title=\"Suggest charts\"\n",
              "                style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "      </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "      <script>\n",
              "        async function quickchart(key) {\n",
              "          const quickchartButtonEl =\n",
              "            document.querySelector('#' + key + ' button');\n",
              "          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "          quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "          try {\n",
              "            const charts = await google.colab.kernel.invokeFunction(\n",
              "                'suggestCharts', [key], {});\n",
              "          } catch (error) {\n",
              "            console.error('Error during call to suggestCharts:', error);\n",
              "          }\n",
              "          quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "          quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "        }\n",
              "        (() => {\n",
              "          let quickchartButtonEl =\n",
              "            document.querySelector('#df-24c01f7d-a426-4b65-b97a-ca94a92124cb button');\n",
              "          quickchartButtonEl.style.display =\n",
              "            google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "        })();\n",
              "      </script>\n",
              "    </div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "df"
            }
          },
          "metadata": {},
          "execution_count": 200
        }
      ],
      "source": [
        "import pandas as pd\n",
        "df = pd.read_csv('/content/sample_data/mnist_test.csv')\n",
        "df.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 201,
      "metadata": {
        "id": "_G0xyLzqxSrg"
      },
      "outputs": [],
      "source": [
        "# train test split\n",
        "X = df.iloc[:,1:].values\n",
        "y = df.iloc[:,0].values\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 202,
      "metadata": {
        "id": "YbgpwszCxqlD"
      },
      "outputs": [],
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 203,
      "metadata": {
        "id": "cHjppFxZx_On"
      },
      "outputs": [],
      "source": [
        "# scaling the feature\n",
        "X_train = X_train / 255.0\n",
        "X_test = X_test / 255.0"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 204,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KyMGnwYHzKVO",
        "outputId": "3e9c7e1c-b560-4ced-8717-1472f53ea0a6"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[0., 0., 0., ..., 0., 0., 0.],\n",
              "       [0., 0., 0., ..., 0., 0., 0.],\n",
              "       [0., 0., 0., ..., 0., 0., 0.],\n",
              "       ...,\n",
              "       [0., 0., 0., ..., 0., 0., 0.],\n",
              "       [0., 0., 0., ..., 0., 0., 0.],\n",
              "       [0., 0., 0., ..., 0., 0., 0.]])"
            ]
          },
          "metadata": {},
          "execution_count": 204
        }
      ],
      "source": [
        "X_train"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 205,
      "metadata": {
        "id": "x_ewUQf0yiXV"
      },
      "outputs": [],
      "source": [
        "from torch.utils.data import Dataset\n",
        "\n",
        "# create CustomDataset class\n",
        "class CustomDataset(Dataset):\n",
        "  def __init__(self, features, labels):\n",
        "    self.features = torch.tensor(features, dtype=torch.float32)\n",
        "    self.labels =torch.tensor(labels, dtype=torch.long)\n",
        "\n",
        "  def __len__(self):\n",
        "    return len(self.features)\n",
        "\n",
        "  def __getitem__(self, index):\n",
        "    return self.features[index], self.labels[index]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 206,
      "metadata": {
        "id": "JenpkLDi24fL"
      },
      "outputs": [],
      "source": [
        "# create train_dataset object\n",
        "train_dataset = CustomDataset(X_train, y_train)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 207,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ndGBMAB13FjE",
        "outputId": "187ab06b-1ed8-478f-8c5c-05f12644a717"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.2235, 0.7765, 1.0000, 1.0000, 1.0000, 0.6667, 0.2235, 0.7765,\n",
              "         0.2235, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.5529, 1.0000, 1.0000, 1.0000, 0.8863, 1.0000, 1.0000, 1.0000,\n",
              "         1.0000, 1.0000, 0.3373, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.6667, 1.0000, 1.0000, 0.6667, 0.1137, 0.0000, 0.0000, 0.8863,\n",
              "         1.0000, 1.0000, 1.0000, 0.6667, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.5529, 1.0000, 0.8863, 0.5529, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.3373, 0.7765, 1.0000, 1.0000, 0.4471, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.3373, 1.0000, 1.0000, 0.3373, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.1137, 0.8863, 1.0000, 1.0000, 0.6667, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.7765, 1.0000, 0.5529, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.1137, 0.6667, 1.0000, 1.0000, 0.8863, 0.1137, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.3373, 1.0000, 1.0000, 0.1137, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.5529, 1.0000, 1.0000, 0.8863, 0.2235, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.3373, 1.0000, 1.0000, 0.0000, 0.0000, 0.1137,\n",
              "         0.2235, 0.4471, 0.7765, 1.0000, 1.0000, 0.8863, 0.1137, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.3373, 1.0000, 1.0000, 1.0000, 1.0000,\n",
              "         1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.7765, 0.1137, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.6667, 1.0000, 1.0000,\n",
              "         1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.6667, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1137, 0.4471,\n",
              "         0.6667, 0.5529, 0.4471, 1.0000, 1.0000, 0.7765, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.1137, 0.6667, 1.0000, 1.0000, 0.1137, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.6667, 1.0000, 1.0000, 0.5529, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.4471, 1.0000, 1.0000, 0.6667, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.2235, 1.0000, 1.0000, 0.7765, 0.1137, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.1137, 0.8863, 1.0000, 0.8863, 0.1137, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.2235, 1.0000, 1.0000, 1.0000, 0.3373, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.2235, 0.8863, 1.0000, 1.0000, 0.6667, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.6667, 1.0000, 1.0000, 0.6667, 0.1137,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4471, 0.8863, 0.3373, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000]),\n",
              " tensor(9))"
            ]
          },
          "metadata": {},
          "execution_count": 207
        }
      ],
      "source": [
        "train_dataset[0]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 208,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "c08svNEd3uFw",
        "outputId": "9b881ddd-4d18-4e46-eec7-0bb13fe9284e"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0980, 0.9804, 0.0863, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.4039, 0.9961, 0.2118, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0118, 0.8471, 0.9961, 0.0863, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0431, 0.0118, 0.0000, 0.0000, 0.0000, 0.1922, 0.9961, 0.9961, 0.0863,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.2980, 0.8471, 0.6235, 0.0000, 0.0000, 0.0000, 0.3725, 0.9961, 0.9961,\n",
              "         0.0863, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.1765, 0.9176, 0.9961, 0.5020, 0.0000, 0.0000, 0.0000, 0.3725, 0.9961,\n",
              "         0.9882, 0.0824, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0314, 0.8235, 0.9961, 0.8039, 0.0235, 0.0000, 0.0000, 0.0000, 0.6078,\n",
              "         0.9961, 0.7333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.1882, 0.9961, 0.9961, 0.4980, 0.0000, 0.0000, 0.0000, 0.0471,\n",
              "         0.8980, 0.9961, 0.6353, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.4431, 0.9961, 0.9961, 0.2863, 0.0000, 0.0000, 0.0000,\n",
              "         0.0706, 0.9961, 0.9961, 0.3843, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.4431, 0.9961, 0.9961, 0.0157, 0.0000, 0.0000,\n",
              "         0.0000, 0.3804, 0.9961, 0.9961, 0.0745, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.4431, 0.9961, 0.9961, 0.0588, 0.0000,\n",
              "         0.0000, 0.0000, 0.4667, 0.9961, 0.9961, 0.1216, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2941, 0.9961, 0.9961, 0.5333,\n",
              "         0.0078, 0.0000, 0.0000, 0.7725, 0.9961, 0.9961, 0.7333, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0275, 0.6784, 0.9961,\n",
              "         0.9961, 0.6588, 0.3725, 0.4157, 0.9373, 0.9961, 0.9961, 0.3529, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1412,\n",
              "         0.8980, 0.9961, 0.9961, 0.9961, 0.9961, 0.9961, 0.9961, 0.8980, 0.0235,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.1412, 0.4863, 0.7961, 0.9961, 0.8118, 0.9961, 0.9961, 0.5882,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0196, 0.0863, 0.1412, 0.9961, 0.9961,\n",
              "         0.4588, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1255, 0.9961,\n",
              "         0.9961, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1255,\n",
              "         0.9961, 0.9961, 0.2980, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.1255, 0.9961, 0.9804, 0.0196, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0392, 0.7137, 0.9922, 0.2157, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,\n",
              "         0.0000]),\n",
              " tensor(4))"
            ]
          },
          "metadata": {},
          "execution_count": 208
        }
      ],
      "source": [
        "# create test_dataset object\n",
        "test_dataset = CustomDataset(X_test, y_test)\n",
        "\n",
        "# Define DataLoaders\n",
        "batch_size = 32\n",
        "train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)\n",
        "test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)\n",
        "\n",
        "# Verify a sample from test_dataset\n",
        "test_dataset[0]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 209,
      "metadata": {
        "id": "3O-MmZWJ5v3V"
      },
      "outputs": [],
      "source": [
        "# define NN class\n",
        "class myNN(nn.Module):\n",
        "\n",
        "  def __init__(self, num_features):\n",
        "\n",
        "    super().__init__()\n",
        "    self.model = nn.Sequential(\n",
        "      nn.Linear(num_features, 128),\n",
        "      nn.BatchNorm1d(128),\n",
        "      nn.ReLU(),\n",
        "      nn.Dropout(p=0.3),\n",
        "      nn.Linear(128, 64),\n",
        "      nn.BatchNorm1d(64),\n",
        "      nn.ReLU(),\n",
        "      nn.Dropout(p=0.3),\n",
        "      nn.Linear(64, 10)\n",
        "    )\n",
        "\n",
        "  def forward(self, x):\n",
        "    return self.model(x)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 210,
      "metadata": {
        "id": "k8odq96n7pdJ"
      },
      "outputs": [],
      "source": [
        "# set learning rate and epoch\n",
        "epochs = 100\n",
        "learning_rate = 0.1"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 211,
      "metadata": {
        "id": "bB2AepLh7wxM"
      },
      "outputs": [],
      "source": [
        "# instatiate the model\n",
        "model = myNN(X_train.shape[1])\n",
        "\n",
        "# loss function\n",
        "criterion = nn.CrossEntropyLoss()\n",
        "\n",
        "# optimizer\n",
        "optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 212,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "F63FAf5t_zfW",
        "outputId": "7c1c5192-364e-4c18-b12f-3484739debec"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "250"
            ]
          },
          "metadata": {},
          "execution_count": 212
        }
      ],
      "source": [
        "len(train_loader)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 213,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "97zU_w997wad",
        "outputId": "f705cb1f-a4d8-47da-b1c8-d5707c3e83e7"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch:1, Loss: 0.6686292321681976\n",
            "Epoch:2, Loss: 0.38270272135734557\n",
            "Epoch:3, Loss: 0.2950041326880455\n",
            "Epoch:4, Loss: 0.2688759104311466\n",
            "Epoch:5, Loss: 0.2610620363205671\n",
            "Epoch:6, Loss: 0.20506945676356553\n",
            "Epoch:7, Loss: 0.19891360525786878\n",
            "Epoch:8, Loss: 0.18701024379581213\n",
            "Epoch:9, Loss: 0.170578038290143\n",
            "Epoch:10, Loss: 0.158534636836499\n",
            "Epoch:11, Loss: 0.15336188046634197\n",
            "Epoch:12, Loss: 0.14591550560295583\n",
            "Epoch:13, Loss: 0.1460287937372923\n",
            "Epoch:14, Loss: 0.11867779667116701\n",
            "Epoch:15, Loss: 0.11655353232659399\n",
            "Epoch:16, Loss: 0.11884785305149853\n",
            "Epoch:17, Loss: 0.10669241434708238\n",
            "Epoch:18, Loss: 0.10620617379248143\n",
            "Epoch:19, Loss: 0.11237544057331979\n",
            "Epoch:20, Loss: 0.10504347890801728\n",
            "Epoch:21, Loss: 0.09580708670057356\n",
            "Epoch:22, Loss: 0.09194096704944968\n",
            "Epoch:23, Loss: 0.09785059397481381\n",
            "Epoch:24, Loss: 0.09052821787539869\n",
            "Epoch:25, Loss: 0.0820261666290462\n",
            "Epoch:26, Loss: 0.0885783860301599\n",
            "Epoch:27, Loss: 0.08403108734916896\n",
            "Epoch:28, Loss: 0.08783208245318383\n",
            "Epoch:29, Loss: 0.08421094013936818\n",
            "Epoch:30, Loss: 0.08167089838627725\n",
            "Epoch:31, Loss: 0.07488962566480041\n",
            "Epoch:32, Loss: 0.0813575523821637\n",
            "Epoch:33, Loss: 0.07162543411832303\n",
            "Epoch:34, Loss: 0.06507485031336546\n",
            "Epoch:35, Loss: 0.06490284679457545\n",
            "Epoch:36, Loss: 0.06780554205505178\n",
            "Epoch:37, Loss: 0.07249254016019403\n",
            "Epoch:38, Loss: 0.06837007207237183\n",
            "Epoch:39, Loss: 0.06367576697282493\n",
            "Epoch:40, Loss: 0.061545233607757835\n",
            "Epoch:41, Loss: 0.05819261626037769\n",
            "Epoch:42, Loss: 0.055570337516255675\n",
            "Epoch:43, Loss: 0.054703038852661846\n",
            "Epoch:44, Loss: 0.0546973302969709\n",
            "Epoch:45, Loss: 0.06459297337336466\n",
            "Epoch:46, Loss: 0.06291352988639846\n",
            "Epoch:47, Loss: 0.05538926929957234\n",
            "Epoch:48, Loss: 0.05491171554056928\n",
            "Epoch:49, Loss: 0.05684111515944824\n",
            "Epoch:50, Loss: 0.0607409654436633\n",
            "Epoch:51, Loss: 0.0608536745137535\n",
            "Epoch:52, Loss: 0.060462154678069056\n",
            "Epoch:53, Loss: 0.05866386686544865\n",
            "Epoch:54, Loss: 0.058826266606803984\n",
            "Epoch:55, Loss: 0.05863993841502815\n",
            "Epoch:56, Loss: 0.05533233598666266\n",
            "Epoch:57, Loss: 0.05460218086559326\n",
            "Epoch:58, Loss: 0.05864652799256146\n",
            "Epoch:59, Loss: 0.049176180402748286\n",
            "Epoch:60, Loss: 0.05053029135381803\n",
            "Epoch:61, Loss: 0.047229974222835154\n",
            "Epoch:62, Loss: 0.04599049201840535\n",
            "Epoch:63, Loss: 0.05102891273796559\n",
            "Epoch:64, Loss: 0.059853313612286005\n",
            "Epoch:65, Loss: 0.04955102764163166\n",
            "Epoch:66, Loss: 0.048164740544278176\n",
            "Epoch:67, Loss: 0.044561575708910825\n",
            "Epoch:68, Loss: 0.053188973671756686\n",
            "Epoch:69, Loss: 0.04373079362232238\n",
            "Epoch:70, Loss: 0.049494079897645864\n",
            "Epoch:71, Loss: 0.050863177294842896\n",
            "Epoch:72, Loss: 0.0427653620170895\n",
            "Epoch:73, Loss: 0.04569593657646328\n",
            "Epoch:74, Loss: 0.04736582716344856\n",
            "Epoch:75, Loss: 0.0426250114236027\n",
            "Epoch:76, Loss: 0.03959964449936524\n",
            "Epoch:77, Loss: 0.04825845818547532\n",
            "Epoch:78, Loss: 0.04570327917812392\n",
            "Epoch:79, Loss: 0.04038094543013722\n",
            "Epoch:80, Loss: 0.03743610271264333\n",
            "Epoch:81, Loss: 0.036336870879284104\n",
            "Epoch:82, Loss: 0.04572136657498777\n",
            "Epoch:83, Loss: 0.044554670805577186\n",
            "Epoch:84, Loss: 0.03777513907942921\n",
            "Epoch:85, Loss: 0.04648944690183271\n",
            "Epoch:86, Loss: 0.03766196619020775\n",
            "Epoch:87, Loss: 0.040625115114729854\n",
            "Epoch:88, Loss: 0.03597093960829079\n",
            "Epoch:89, Loss: 0.04719071347173304\n",
            "Epoch:90, Loss: 0.0431760856944602\n",
            "Epoch:91, Loss: 0.04584159775532316\n",
            "Epoch:92, Loss: 0.042380455594975504\n",
            "Epoch:93, Loss: 0.04071473558899015\n",
            "Epoch:94, Loss: 0.04609146722045261\n",
            "Epoch:95, Loss: 0.041857291731750595\n",
            "Epoch:96, Loss: 0.045030185651034116\n",
            "Epoch:97, Loss: 0.04181439650221728\n",
            "Epoch:98, Loss: 0.042741042677662336\n",
            "Epoch:99, Loss: 0.0360375844280934\n",
            "Epoch:100, Loss: 0.04365045154676773\n"
          ]
        }
      ],
      "source": [
        "# training loop\n",
        "for epoch in range(epochs):\n",
        "\n",
        "  total_epoch_loss = 0.0\n",
        "  for batch_features, batch_labels in train_loader:\n",
        "\n",
        "\n",
        "    # forward pass\n",
        "    outputs = model(batch_features)\n",
        "\n",
        "\n",
        "    # calculate loss\n",
        "    loss = criterion(outputs, batch_labels)\n",
        "\n",
        "    # back pass\n",
        "    optimizer.zero_grad()\n",
        "    loss.backward()\n",
        "\n",
        "    # update grads\n",
        "    optimizer.step()\n",
        "\n",
        "    total_epoch_loss = total_epoch_loss + loss.item()\n",
        "\n",
        "  avg_loss = total_epoch_loss / len(train_loader)\n",
        "  print(f'Epoch:{epoch+1}, Loss: {avg_loss}')"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.eval()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fGmBI010gSDp",
        "outputId": "750e24a2-3de4-4125-bab0-e07a581e74ad"
      },
      "execution_count": 214,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "myNN(\n",
              "  (model): Sequential(\n",
              "    (0): Linear(in_features=784, out_features=128, bias=True)\n",
              "    (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
              "    (2): ReLU()\n",
              "    (3): Dropout(p=0.3, inplace=False)\n",
              "    (4): Linear(in_features=128, out_features=64, bias=True)\n",
              "    (5): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
              "    (6): ReLU()\n",
              "    (7): Dropout(p=0.3, inplace=False)\n",
              "    (8): Linear(in_features=64, out_features=10, bias=True)\n",
              "  )\n",
              ")"
            ]
          },
          "metadata": {},
          "execution_count": 214
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 215,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3yGllNNQ8Lka",
        "outputId": "6a8b6406-a2c8-418a-ea39-454d8834d2a6"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "myNN(\n",
              "  (model): Sequential(\n",
              "    (0): Linear(in_features=784, out_features=128, bias=True)\n",
              "    (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
              "    (2): ReLU()\n",
              "    (3): Dropout(p=0.3, inplace=False)\n",
              "    (4): Linear(in_features=128, out_features=64, bias=True)\n",
              "    (5): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
              "    (6): ReLU()\n",
              "    (7): Dropout(p=0.3, inplace=False)\n",
              "    (8): Linear(in_features=64, out_features=10, bias=True)\n",
              "  )\n",
              ")"
            ]
          },
          "metadata": {},
          "execution_count": 215
        }
      ],
      "source": [
        "# set model to eval mode\n",
        "model.eval()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 216,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Tje_RZNwCb6_",
        "outputId": "34a20aa7-7c31-4032-dce6-e303a7da1e8a"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "63"
            ]
          },
          "metadata": {},
          "execution_count": 216
        }
      ],
      "source": [
        "len(test_loader)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 217,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "v3cBUX_M8QSU",
        "outputId": "afe0a158-e9b9-4cee-d5d7-3b34f3d401b2"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy of the network on the 10000 test images: 96.60%\n"
          ]
        }
      ],
      "source": [
        "import torch\n",
        "# evaluation code\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "model.to(device)\n",
        "with torch.no_grad():\n",
        "    correct = 0\n",
        "    total = 0\n",
        "    for batch_features, batch_labels in test_loader:\n",
        "        batch_features = batch_features.to(device)\n",
        "        batch_labels = batch_labels.to(device)\n",
        "        outputs = model(batch_features)\n",
        "        _, predicted = torch.max(outputs.data, 1)\n",
        "        total += batch_labels.size(0)\n",
        "        correct += (predicted == batch_labels).sum().item()\n",
        "\n",
        "    accuracy = 100 * correct / total\n",
        "    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f}%')"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# evaluation on training data\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "model.to(device)\n",
        "total = 0\n",
        "correct = 0\n",
        "with torch.no_grad():\n",
        "  for batch_features, batch_labels in train_loader:\n",
        "    # move data to gpu\n",
        "    batch_features = batch_features.to(device)\n",
        "    batch_labels = batch_labels.to(device)\n",
        "    outputs = model(batch_features)\n",
        "    _, predicted = torch.max(outputs.data, 1)\n",
        "    total = total + batch_labels.shape[0]\n",
        "    correct = correct + (predicted == batch_labels).sum().item()\n",
        "    print(correct/total)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "G8pHFd1Kg1MW",
        "outputId": "524f345e-a003-432a-989f-b01b566d7cfe"
      },
      "execution_count": 218,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n",
            "1.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from tqdm import tqdm\n",
        "\n",
        "# ... (device and model setup)\n",
        "\n",
        "total = 0\n",
        "correct = 0\n",
        "\n",
        "with torch.no_grad():\n",
        "    # Wrap your loader with tqdm for a progress bar\n",
        "    pbar = tqdm(train_loader, desc=\"Evaluating\")\n",
        "\n",
        "    for batch_features, batch_labels in pbar:\n",
        "        batch_features = batch_features.to(device)\n",
        "        batch_labels = batch_labels.to(device)\n",
        "\n",
        "        outputs = model(batch_features)\n",
        "        _, predicted = torch.max(outputs, 1)\n",
        "\n",
        "        total += batch_labels.size(0)\n",
        "        correct += (predicted == batch_labels).sum().item()\n",
        "\n",
        "        # Optional: Update the progress bar with current batch accuracy\n",
        "        current_acc = 100 * correct / total\n",
        "        pbar.set_postfix(accuracy=f\"{current_acc:.2f}%\")\n",
        "\n",
        "print(f\"\\nFinal Training Accuracy: {100 * correct / total:.2f}%\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Rm33kLGFj5fF",
        "outputId": "3994d19a-4061-485f-9ea3-d88c03517bfb"
      },
      "execution_count": 219,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Evaluating: 100%|██████████| 250/250 [00:00<00:00, 456.44it/s, accuracy=100.00%]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Final Training Accuracy: 100.00%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def evaluate(loader, dataset_name):\n",
        "    model.eval() # Set model to evaluation mode\n",
        "    total = 0\n",
        "    correct = 0\n",
        "\n",
        "    with torch.no_grad():\n",
        "        pbar = tqdm(loader, desc=f\"Evaluating {dataset_name}\")\n",
        "        for batch_features, batch_labels in pbar:\n",
        "            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)\n",
        "\n",
        "            outputs = model(batch_features)\n",
        "            _, predicted = torch.max(outputs, 1)\n",
        "\n",
        "            total += batch_labels.size(0)\n",
        "            correct += (predicted == batch_labels).sum().item()\n",
        "\n",
        "            pbar.set_postfix(acc=f\"{100 * correct / total:.2f}%\")\n",
        "\n",
        "    return 100 * correct / total\n",
        "\n",
        "# Run evaluation\n",
        "train_acc = evaluate(train_loader, \"Train\")\n",
        "test_acc = evaluate(test_loader, \"Test\")\n",
        "\n",
        "print(f\"\\nFinal Results:\\nTrain Accuracy: {train_acc:.2f}%\\nTest Accuracy: {test_acc:.2f}%\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "la9xlz94kFlw",
        "outputId": "59280f96-fd25-430c-fbd5-05a34ae129fb"
      },
      "execution_count": 220,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Evaluating Train: 100%|██████████| 250/250 [00:00<00:00, 410.49it/s, acc=100.00%]\n",
            "Evaluating Test: 100%|██████████| 63/63 [00:00<00:00, 406.28it/s, acc=96.60%]"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Final Results:\n",
            "Train Accuracy: 100.00%\n",
            "Test Accuracy: 96.60%\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 221,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XE64gCDMEVN_",
        "outputId": "4d4bb3b1-fad7-4700-eae2-5c93248bb1b4"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Using Colab cache for faster access to the 'fashionmnist' dataset.\n",
            "Path to dataset files: /kaggle/input/fashionmnist\n"
          ]
        }
      ],
      "source": [
        "import kagglehub\n",
        "\n",
        "# Download latest version\n",
        "path = kagglehub.dataset_download(\"zalando-research/fashionmnist\")\n",
        "\n",
        "print(\"Path to dataset files:\", path)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "import requests\n",
        "from PIL import Image\n",
        "from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification\n",
        "import io # Import io module\n",
        "\n",
        "# --- STEP 1: DOWNLOAD THE IMAGE FROM THE INTERNET ---\n",
        "# This solves the FileNotFoundError by fetching a sample invoice\n",
        "url = \"https://picsum.photos/200/300.jpg\" # Changed to a generic image URL for testing\n",
        "response = requests.get(url)\n",
        "image_data = response.content\n",
        "\n",
        "# Instead of saving to file and then opening, directly open from bytes\n",
        "image = Image.open(io.BytesIO(image_data)).convert(\"RGB\") # Changed this line to open from bytes\n",
        "\n",
        "# Optionally, still save to disk if needed for other purposes, but not required for this specific error fix\n",
        "with open(\"invoice_sample.jpg\", \"wb\") as handler:\n",
        "    handler.write(image_data)\n",
        "print(\"✅ Step 1: 'invoice_sample.jpg' downloaded successfully (and opened from memory).\")\n",
        "\n",
        "# --- STEP 2: INITIALIZE THE MODEL ---\n",
        "print(\"⏳ Step 2: Loading AI Model (this may take a minute)...\")\n",
        "processor = LayoutLMv3Processor.from_pretrained(\"microsoft/layoutlmv3-base\", apply_ocr=True)\n",
        "model = LayoutLMv3ForTokenClassification.from_pretrained(\"microsoft/layoutlmv3-base\", num_labels=4)\n",
        "\n",
        "# --- STEP 3: RUN THE INFERENCE ---\n",
        "# The 'image' variable already holds the PIL Image object from the BytesIO stream\n",
        "encoding = processor(image, return_tensors=\"pt\")\n",
        "\n",
        "with torch.no_grad():\n",
        "    outputs = model(**encoding)\n",
        "\n",
        "# Get predictions\n",
        "predictions = outputs.logits.argmax(-1).squeeze().tolist()\n",
        "print(f\"✅ Step 3: Success! Detected {len(predictions)} document elements.\")\n",
        "\n",
        "# Show the image to confirm it worked\n",
        "# Note: image.show() typically opens a viewer, which might not work directly in Colab.\n",
        "# For visual confirmation in Colab, you might use IPython.display.Image or matplotlib.\n",
        "# Keeping it as is for now as it doesn't cause a functional error after the fix.\n",
        "image.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "94fHqIYbo3LU",
        "outputId": "fac998e7-ebc7-4fc5-a3f0-e80a0ec41057"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Step 1: 'invoice_sample.jpg' downloaded successfully (and opened from memory).\n",
            "⏳ Step 2: Loading AI Model (this may take a minute)...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Some weights of LayoutLMv3ForTokenClassification were not initialized from the model checkpoint at microsoft/layoutlmv3-base and are newly initialized: ['classifier.bias', 'classifier.weight']\n",
            "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n",
            "/usr/local/lib/python3.12/dist-packages/transformers/modeling_utils.py:1621: FutureWarning: The `device` argument is deprecated and will be removed in v5 of Transformers.\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Step 3: Success! Detected 2 document elements.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Install pytesseract for OCR functionality\n",
        "!pip install pytesseract\n",
        "\n",
        "import torch\n",
        "import requests\n",
        "from PIL import Image\n",
        "from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification\n",
        "import io # Import io module for BytesIO\n",
        "\n",
        "# 1. Download sample data\n",
        "# Using a reliable URL for a sample invoice image\n",
        "url = \"https://tesseract.projectnaptha.com/img/eng_bw.png\" # Using a stable image URL from Tesseract project\n",
        "\n",
        "try:\n",
        "    response = requests.get(url)\n",
        "    response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)\n",
        "    image_bytes = response.content\n",
        "    img = Image.open(io.BytesIO(image_bytes)).convert(\"RGB\")\n",
        "    img.save(\"invoice_sample.png\") # Save as .png\n",
        "    print(\"✅ Image downloaded and saved as 'invoice_sample.png'\")\n",
        "except requests.exceptions.RequestException as e:\n",
        "    print(f\"❌ Error downloading image: {e}\")\n",
        "    # Fallback: create a blank image to avoid crashing the rest of the code\n",
        "    img = Image.new('RGB', (224, 224), color = (255, 255, 255))\n",
        "    print(\"⚠️ Using a blank placeholder image due to download error.\")\n",
        "except Exception as e:\n",
        "    print(f\"❌ Error processing image: {e}\")\n",
        "    # Fallback: create a blank image\n",
        "    img = Image.new('RGB', (224, 224), color = (255, 255, 255))\n",
        "    print(\"⚠️ Using a blank placeholder image due to processing error.\")\n",
        "\n",
        "# 2. Load Model & Processor\n",
        "processor = LayoutLMv3Processor.from_pretrained(\"microsoft/layoutlmv3-base\", apply_ocr=True)\n",
        "model = LayoutLMv3ForTokenClassification.from_pretrained(\"microsoft/layoutlmv3-base\", num_labels=4)\n",
        "\n",
        "# 3. Inference\n",
        "encoding = processor(img, return_tensors=\"pt\")\n",
        "with torch.no_grad():\n",
        "    outputs = model(**encoding)\n",
        "\n",
        "predictions = outputs.logits.argmax(-1).squeeze().tolist()\n",
        "print(f\"Project 1 Success: Detected {len(predictions)} document elements.\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8Qh_IFQuq0LJ",
        "outputId": "72e42d3f-caf1-45d0-8aef-b4b96bfda494"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: pytesseract in /usr/local/lib/python3.12/dist-packages (0.3.13)\n",
            "Requirement already satisfied: packaging>=21.3 in /usr/local/lib/python3.12/dist-packages (from pytesseract) (25.0)\n",
            "Requirement already satisfied: Pillow>=8.0.0 in /usr/local/lib/python3.12/dist-packages (from pytesseract) (11.3.0)\n",
            "✅ Image downloaded and saved as 'invoice_sample.png'\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Some weights of LayoutLMv3ForTokenClassification were not initialized from the model checkpoint at microsoft/layoutlmv3-base and are newly initialized: ['classifier.bias', 'classifier.weight']\n",
            "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n",
            "/usr/local/lib/python3.12/dist-packages/transformers/modeling_utils.py:1621: FutureWarning: The `device` argument is deprecated and will be removed in v5 of Transformers.\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Project 1 Success: Detected 94 document elements.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# https://pypi.python.org/pypi/pydot\n",
        "!apt-get -qq install -y graphviz && pip install pydot\n",
        "import pydot\n",
        "\n",
        "import torch\n",
        "import requests\n",
        "import io\n",
        "from PIL import Image\n",
        "from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification\n",
        "\n",
        "# 1. Reliable Download with User-Agent\n",
        "url = \"https://tesseract.projectnaptha.com/img/eng_bw.png\" # Using a stable image URL from Tesseract project\n",
        "headers = {\"User-Agent\": \"Mozilla/5.0\"}\n",
        "response = requests.get(url, headers=headers)\n",
        "\n",
        "if response.status_code == 200:\n",
        "    # Use io.BytesIO to convert raw web data into a format PIL can read\n",
        "    img = Image.open(io.BytesIO(response.content)).convert(\"RGB\")\n",
        "    img.save(\"invoice_sample.png\")\n",
        "    print(\"✅ Image downloaded successfully!\")\n",
        "else:\n",
        "    print(f\"❌ Error: Could not download image. Status code: {response.status_code}\")\n",
        "    # Fallback to a blank image to allow the rest of the code to run\n",
        "    img = Image.new('RGB', (224, 224), color = (255, 255, 255))\n",
        "    print(\"⚠️ Using a blank placeholder image due to download error.\")\n",
        "\n",
        "# 2. Run the AI Model\n",
        "processor = LayoutLMv3Processor.from_pretrained(\"microsoft/layoutlmv3-base\", apply_ocr=True)\n",
        "model = LayoutLMv3ForTokenClassification.from_pretrained(\"microsoft/layoutlmv3-base\", num_labels=4)\n",
        "\n",
        "encoding = processor(img, return_tensors=\"pt\")\n",
        "with torch.no_grad():\n",
        "    outputs = model(**encoding)\n",
        "\n",
        "print(f\"🚀 Success! Detected {outputs.logits.shape[1]} document features.\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "akpzHVwStJwv",
        "outputId": "b3fddcd0-e8d2-4730-91d4-b9451c7ae65c"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: pydot in /usr/local/lib/python3.12/dist-packages (4.0.1)\n",
            "Requirement already satisfied: pyparsing>=3.1.0 in /usr/local/lib/python3.12/dist-packages (from pydot) (3.2.5)\n",
            "✅ Image downloaded successfully!\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Some weights of LayoutLMv3ForTokenClassification were not initialized from the model checkpoint at microsoft/layoutlmv3-base and are newly initialized: ['classifier.bias', 'classifier.weight']\n",
            "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n",
            "/usr/local/lib/python3.12/dist-packages/transformers/modeling_utils.py:1621: FutureWarning: The `device` argument is deprecated and will be removed in v5 of Transformers.\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "🚀 Success! Detected 94 document features.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from PIL import ImageDraw\n",
        "\n",
        "# 1. Get the original image and create a drawing object\n",
        "draw = ImageDraw.Draw(img)\n",
        "\n",
        "# 2. Get the boxes from the encoding (unnormalize them to match image size)\n",
        "# LayoutLMv3 uses a 0-1000 scale, so we scale them back to the actual pixels\n",
        "width, height = img.size\n",
        "boxes = encoding['bbox'][0].tolist()\n",
        "\n",
        "for box in boxes:\n",
        "    # Skip dummy boxes [0,0,0,0]\n",
        "    if box == [0, 0, 0, 0]:\n",
        "        continue\n",
        "\n",
        "    # Unnormalize: (x / 1000) * width\n",
        "    unnorm_box = [\n",
        "        box[0] * width / 1000,\n",
        "        box[1] * height / 1000,\n",
        "        box[2] * width / 1000,\n",
        "        box[3] * height / 1000\n",
        "    ]\n",
        "\n",
        "    # Draw the rectangle\n",
        "    draw.rectangle(unnorm_box, outline=\"red\", width=2)\n",
        "\n",
        "# 3. Show the final result\n",
        "print(\"✨ Visualization Ready! Red boxes show detected text areas.\")\n",
        "display(img)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 518
        },
        "id": "k2d70dbfydgP",
        "outputId": "e6391081-a5c5-40d9-ccf0-3f9bcb9179f4"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✨ Visualization Ready! Red boxes show detected text areas.\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<PIL.Image.Image image mode=RGB size=1486x668>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAABc4AAAKcCAIAAABe874EAAEAAElEQVR4Aeyd65rdNs6sv8yz7/+Ws9FmDMPEgeBJoqTyjwxFAoXCS0prtdLO/PPvv//+H/6AAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAisIPC/FSLQAAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQ+CGAVy04ByAAAiAAAiAAAiAAAiAAAiAAAiAAAiCwjABetSxDCSEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQwKsWnAEQAAEQAAEQAAEQAAEQAAEQAAEQAAEQWEYAr1qWoYQQCIAACIAACIAACIAACIAACIAACIAACOBVC84ACIAACIAACIAACIAACIAACIAACIAACCwjgFcty1BCCARAAARAAARAAARAAARAAARAAARAAATwqgVnAARAAARAAARAAARAAARAAARAAARAAASWEcCrlmUoIQQCIAACIAACIAACIAACIAACIAACIAACeNWCMwACIAACIAACIAACIAACIAACIAACIAACywjgVcsylBACARAAARAAARAAARAAARAAARAAARAAAbxqwRkAARAAARAAARAAARAAARAAARAAARAAgWUE8KplGUoIgQAIgAAIgAAIgAAIgAAIgAAIgAAIgABeteAMgAAIgAAIgAAIgAAIgAAIgAAIgAAIgMAyAnjVsgwlhEAABEAABEAABEAABEAABEAABEAABEAAr1pwBkAABEAABEAABEAABEAABEAABEAABEBgGQG8almGEkIgAAIgAAIgAAIgAAIgAAIgAAIgAAIggFctOAMgAAIgAAIgAAIgAAIgAAIgAAIgAAIgsIwAXrUsQwkhEAABEAABEAABEAABEAABEAABEAABEMCrFpwBEAABEAABEAABEAABEAABEAABEAABEFhGAK9alqGEEAiAAAiAAAiAAAiAAAiAAAiAAAiAAAjgVQvOAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAgsI4BXLctQQggEQAAEQAAEQAAEQAAEQAAEQAAEQAAE8KoFZwAEQAAEQAAEQAAEQAAEQAAEQAAEQAAElhHAq5ZlKCEEAiAAAiAAAiAAAiAAAiAAAiAAAiAAAnjVgjMAAiAAAiAAAiAAAiAAAiAAAiAAAiAAAssI4FXLMpQQAgEQAAEQAAEQAAEQAAEQAAEQAAEQAAG8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\n",
            "image/jpeg": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAKcBc4DASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA4b4qaq2jeGrO8+13ttAt8n2h7GTZKYgjlgpyBnA4B4yBXjGm/FvTZbhl1SfxlbQbCVe11lZ2LZHBVkQAYzznsOOePa/iTY2+p6dodheR+Za3WswQzJuI3IwcMMjkZBPSvCfit8I4fAVlBq2nalJc6fPcC38m4UebGxQsDuXAYHa/ZcfKOeTWHJGU5X8j1XiatDC0vZu1+bou53fhLxZ4Z8Y+IItFsvEnjm3u5kZovtV2oVyo3FQVLYO0E84HB5zgH0P8A4QT/AKmvxR/4Mf8A7GvjWwvrjTNRtr+zk8u6tZUmhfaDtdSCpweDggda+59C1P8Atvw9pmreT5P260iufK3btm9A23OBnGcZwKr2MOxh/aWK/m/Bf5GF/wAIJ/1Nfij/AMGP/wBjR/wgn/U1+KP/AAY//Y11teb/ABz1n+yPhfexK88c2oSx2cbwnGMnewY5HylEdT1zuxjBNHsYdg/tLFfzfgv8jK/tnwN/bH9l/wDC0Na+0f3/AO0z5P3d3+u2eX0/2uvHXitTSLbw5r+qXem6R8Q9dvbu1RXlSDVNw2nurbcOBkAlScEgHBNfJNfT/wAKfg9b+GZ7HxRqN1PNqT2iSRWrwmH7I8ifOGG4lmAYrzjHPGcYPYw7B/aWK/m/Bf5HZ/8ACCf9TX4o/wDBj/8AY1XsfCdnqdnHeWHjXxDd2smdk0GqiRGwSDhgMHBBH4VU+NM2qR/C/VE0q1nmaXaly8Em1oYM5dyByy4G1gP4XJPANeAfBb+2P+FoaX/ZH+19s3b/AC/s+Pn37fw25437M0exh2D+0sV/N+C/yOv+LOva94D8VWul6Xr+rTQS2SXDNdXjswYu64G0qMYQdvWuD/4Wp4w/6DN3/wCBU3/xdfVHj3w5D4p8Fappr2Md3cG3kezRiFK3AU+WVYkbTu4zkDBIPBNfHHhrTYdZ8VaRpdw0iwXt7DbyNGQGCu4UkZBGcH0NHsYdg/tLFfzfgv8AI+vv+EE/6mvxR/4Mf/saP+EE/wCpr8Uf+DH/AOxrraKPYw7B/aWK/m/Bf5HifxYnvPh/odrcWGs+KLq6u5TGk018TBDjBO/ABLEZ2rkZwxz8uD4//wALU8Yf9Bm7/wDAqb/4uvsS+sLPU7OSzv7SC7tZMb4Z4xIjYIIyp4OCAfwr5Y+PGh6XoPj63i0mwgsoZ9PjmkigTYm/fIuQo4XhF6AevUk0exh2D+0sV/N+C/yNz4eeM9d128tGu7/V2ki1a0heQXxMDRyFvkZCd247DzypGQcHG76Tr5r+FFjbx+DbC/WPF1N4utYZH3H5kSPKjHTgyP8An7CvpSlTioykl5GmMqSq0KU57+9+YUUUVseac74v1nUdGs9O/stLVrq8v47NftQYoN4bk7SD1A9e/FZGpat4x0a3W41TUPBtjAzhFkuppolLYJwCxAzgE49jVj4hTw2tv4euLiWOGCLW7Z5JJGCqigOSSTwABzmvmHxX4s134n+LbfzE+aWUW+n2COAkW9gAoJwCxOMscZ9gABhZym9Wen7SNHDU2oRbd73V9mex/wDC7f8AqYfC/wD4Bah/8brs9N1bxjrNu1xpeoeDb6BXKNJazTSqGwDglSRnBBx7ivNYP2Zpmt4muPFccc5QGRI7AuqtjkBjICRnvgZ9BXnfxE+GWqfD68iMsn23TJ8CG+SPYC+MlGXJ2twSOTkcg8MBXsvNmX17/p3D7j6a/wCLh/8AUr/+TFH/ABcP/qV//JisT4GPqkvwvspdSvPtMbSyCyyctFAp2BCcdmV8cnClRnAwPSKPZebD69/07h9xyX/Fw/8AqV//ACYo/wCLh/8AUr/+TFdbRR7LzYfXv+ncPuOS/wCLh/8AUr/+TFcH41+LPiXwHrMOl6paaTNPLbrcK1rHIyhSzLg7nU5yh7ele01x/jr4b6F48sz9uh8nUo4jHbX8ed8XORkZAdc/wn+82CpOaPZebD69/wBO4fceR/8ADR2p/wDQNtP+/Df/AB2vV7C+8danp1tf2cnheS1uokmhfbcjcjAFTg8jII618a1972Fjb6Zp1tYWcfl2trEkMKbidqKAFGTycADrR7LzYfXv+ncPuOa/4uH/ANSv/wCTFH/Fw/8AqV//ACYrraKPZebD69/07h9xyX/Fw/8AqV//ACYo/wCLh/8AUr/+TFdbRR7LzYfXv+ncPuOS/wCLh/8AUr/+TFH/ABcP/qV//Jiutoo9l5sPr3/TuH3HJf8AFw/+pX/8mKP+Lh/9Sv8A+TFdbRR7LzYfXv8Ap3D7jkv+Lh/9Sv8A+TFZfiPX/Gnhbw/ea1qTeGxaWqBn8tbhmYkhVUD1LEDnA55IHNeg1Xv7G31PTrmwvI/MtbqJ4Zk3EbkYEMMjkZBPSj2Xmw+vf9O4fcfPf/DR2p/9A20/78N/8drY8LfGvX/F3iO00OwstMjurrfseeKQINqM5yRIT0U9q5z4mfBGz8I+HJ9f0bVJ5LW18sT214AXO59u5XUAdWT5Sv8AeO7oKw/gNps198VLO4iaMJYW81xKGJyVKGLC8dd0innHAP0J7LzYfXv+ncPuPob/AIuH/wBSv/5MUf8AFw/+pX/8mK62ij2Xmw+vf9O4fccl/wAXD/6lf/yYo/4uH/1K/wD5MV1tFHsvNh9e/wCncPuOS/4uH/1K/wD5MUf8XD/6lf8A8mK62ij2Xmw+vf8ATuH3HJf8XD/6lf8A8mKP+Lh/9Sv/AOTFdbRR7LzYfXv+ncPuOS/4uH/1K/8A5MUf8XD/AOpX/wDJiutoo9l5sPr3/TuH3HJf8XD/AOpX/wDJipvDGs6ze6zrOl60lgJ9P8jDWQfa3mKW/iOegHYd66euS8P/APJQvGP/AG5f+ijUuLjKNm9X+jNoVY16NbmhFcsU00rO/PFfk2dbRRRW55ZU1W+/szSL2/8AL8z7LBJNs3Y3bVJxntnFcxa+KfFN7ZwXdv4K3wTxrJG39qxDcrDIOCM9DW54p/5FDWv+vCf/ANFtR4W/5FDRf+vCD/0WtYy5nPlTtp5Ho0fZU8K6sqak+a2vNta/RoyP+Eg8X/8AQj/+VaL/AAo/4SDxf/0I/wD5Vov8K62in7OX8z/D/Ij63R/58Q++f/yZyX/CQeL/APoR/wDyrRf4Uf8ACQeL/wDoR/8AyrRf4V1tef8Axb8f/wDCC+F/9Dk26zf7o7LMW9VwV3uc8fKGGM5yxXgjOD2cv5n+H+QfW6P/AD4h98//AJMwtZ+OUGgai9hqOjwJdJkOkOpLNsIJBVjGjBWBByp5HpV/w98X18RPbPa6NH9llvorF5RfAtG8nQmMoGxgMQcYO0jOQa+fvAvw41j4gfb/AOybmxh+w+X5n2t3XO/djG1W/uHrjtV/4b6TqWnfEawS+0+7tXgu7ZJlnhZDGzyKyBsjgsoJAPUAkVM4yjG6k/w/yOjC1aFeqqcqEVe+znfbzmz7Aooorc8kqarff2ZpF7f+X5n2WCSbZuxu2qTjPbOK5i18U+Kb2zgu7fwVvgnjWSNv7ViG5WGQcEZ6GtzxT/yKGtf9eE//AKLao/Ds8Nr4I0m4uJY4YItNheSSRgqoojBJJPAAHOaxlzOfKnbTyPRo+yp4V1ZU1J81teba1+jRmf8ACQeL/wDoR/8AyrRf4VT1Xxr4h0PS7jU9T8Ix21nbpvllfVosKPyySTgADkkgDJNdbpurabrNu1xpeoWl9ArlGktZllUNgHBKkjOCDj3FY/jXwVpvjzRodL1Se7hgiuFuFa1dVYsFZcHcrDGHPb0p+zl/M/w/yI+t0f8AnxD75/8AyZ5v/wANG6X/ANAj/wAmW/8AjVdXoHxC1jxTpa6lovhWO7tC5TeuqxqVYdQysoKnocEDgg9CK+WfFXh648KeKNR0O6bfJaSlA+APMQjKPgE43KVOM8Zwea+m/gNpsNj8K7O4iaQvf3E1xKGIwGDmLC8dNsannPJP0B7OX8z/AA/yD63R/wCfEPvn/wDJm/8A8JB4v/6Ef/yrRf4Uf8JB4v8A+hH/APKtF/hXW0Uezl/M/wAP8g+t0f8AnxD75/8AyZyX/CQeL/8AoR//ACrRf4Uf8JB4v/6Ef/yrRf4V1tFHs5fzP8P8g+t0f+fEPvn/APJnJf8ACQeL/wDoR/8AyrRf4VFJ4v16yvNPj1Twp9jgvLuO0Wb+0Uk2s5/uquegJ7dK7KuS8d/8y1/2HrX/ANmqZxlGN1J/h/kdGFq0K9VU5UIq99nO+3nNnW0UUVueSFZP/CU+Hv8AoPaX/wCBkf8AjWtXA/DzQNGvfAum3F3pFhPO/m7pJbZHZsSuBkkZ6ACs5ylzKMfP9P8AM7cPRoujOtWb0cVpbqpPr/hOo/4Snw9/0HtL/wDAyP8Axo/4Snw9/wBB7S//AAMj/wAaP+EW8Pf9AHS//AOP/Cj/AIRbw9/0AdL/APAOP/Cl+98h/wCw/wB/8A/4Snw9/wBB7S//AAMj/wAaP+Ep8Pf9B7S//AyP/Gj/AIRbw9/0AdL/APAOP/Cj/hFvD3/QB0v/AMA4/wDCj975B/sP9/8AAP8AhKfD3/Qe0v8A8DI/8aP+Ep8Pf9B7S/8AwMj/AMaP+EW8Pf8AQB0v/wAA4/8ACj/hFvD3/QB0v/wDj/wo/e+Qf7D/AH/wD/hKfD3/AEHtL/8AAyP/ABo/4Snw9/0HtL/8DI/8aP8AhFvD3/QB0v8A8A4/8KP+EW8Pf9AHS/8AwDj/AMKP3vkH+w/3/wAA/wCEp8Pf9B7S/wDwMj/xo/4Snw9/0HtL/wDAyP8Axo/4Rbw9/wBAHS//AADj/wAKP+EW8Pf9AHS//AOP/Cj975B/sP8Af/AP+Ep8Pf8AQe0v/wADI/8AGj/hKfD3/Qe0v/wMj/xo/wCEW8Pf9AHS/wDwDj/wrzf/AIWR8Kf+Eo/sT+y7Lb5vk/2j9hh+ybsdd+c7c8btu3vnb81H73yD/Yf7/wCB6R/wlPh7/oPaX/4GR/40f8JT4e/6D2l/+Bkf+NH/AAi3h7/oA6X/AOAcf+FH/CLeHv8AoA6X/wCAcf8AhR+98g/2H+/+Af8ACU+Hv+g9pf8A4GR/40f8JT4e/wCg9pf/AIGR/wCNH/CLeHv+gDpf/gHH/hR/wi3h7/oA6X/4Bx/4UfvfIP8AYf7/AOAf8JT4e/6D2l/+Bkf+NH/CU+Hv+g9pf/gZH/jR/wAIt4e/6AOl/wDgHH/hR/wi3h7/AKAOl/8AgHH/AIUfvfIP9h/v/gH/AAlPh7/oPaX/AOBkf+NH/CU+Hv8AoPaX/wCBkf8AjR/wi3h7/oA6X/4Bx/4Uf8It4e/6AOl/+Acf+FH73yD/AGH+/wDgH/CU+Hv+g9pf/gZH/jR/wlPh7/oPaX/4GR/40f8ACLeHv+gDpf8A4Bx/4Uf8It4e/wCgDpf/AIBx/wCFH73yD/Yf7/4B/wAJT4e/6D2l/wDgZH/jR/wlPh7/AKD2l/8AgZH/AI0f8It4e/6AOl/+Acf+FH/CLeHv+gDpf/gHH/hR+98g/wBh/v8A4B/wlPh7/oPaX/4GR/40f8JT4e/6D2l/+Bkf+NH/AAi3h7/oA6X/AOAcf+FH/CLeHv8AoA6X/wCAcf8AhR+98g/2H+/+Af8ACU+Hv+g9pf8A4GR/40f8JT4e/wCg9pf/AIGR/wCNH/CLeHv+gDpf/gHH/hR/wi3h7/oA6X/4Bx/4UfvfIP8AYf7/AOAf8JT4e/6D2l/+Bkf+NH/CU+Hv+g9pf/gZH/jR/wAIt4e/6AOl/wDgHH/hR/wi3h7/AKAOl/8AgHH/AIUfvfIP9h/v/gH/AAlPh7/oPaX/AOBkf+NH/CU+Hv8AoPaX/wCBkf8AjR/wi3h7/oA6X/4Bx/4Uf8It4e/6AOl/+Acf+FH73yD/AGH+/wDgH/CU+Hv+g9pf/gZH/jR/wlPh7/oPaX/4GR/40f8ACLeHv+gDpf8A4Bx/4Uf8It4e/wCgDpf/AIBx/wCFH73yD/Yf7/4B/wAJT4e/6D2l/wDgZH/jR/wlPh7/AKD2l/8AgZH/AI0f8It4e/6AOl/+Acf+FH/CLeHv+gDpf/gHH/hR+98g/wBh/v8A4B/wlPh7/oPaX/4GR/40f8JT4e/6D2l/+Bkf+NH/AAi3h7/oA6X/AOAcf+FH/CLeHv8AoA6X/wCAcf8AhR+98g/2H+/+Af8ACU+Hv+g9pf8A4GR/40f8JT4e/wCg9pf/AIGR/wCNH/CLeHv+gDpf/gHH/hR/wi3h7/oA6X/4Bx/4UfvfIP8AYf7/AOAf8JT4e/6D2l/+Bkf+NH/CU+Hv+g9pf/gZH/jR/wAIt4e/6AOl/wDgHH/hR/wi3h7/AKAOl/8AgHH/AIUfvfIP9h/v/gH/AAlPh7/oPaX/AOBkf+NH/CU+Hv8AoPaX/wCBkf8AjR/wi3h7/oA6X/4Bx/4Uf8It4e/6AOl/+Acf+FH73yD/AGH+/wDgH/CU+Hv+g9pf/gZH/jR/wlPh7/oPaX/4GR/40f8ACLeHv+gDpf8A4Bx/4Uf8It4e/wCgDpf/AIBx/wCFH73yD/Yf7/4B/wAJT4e/6D2l/wDgZH/jR/wlPh7/AKD2l/8AgZH/AI18Z+I9U11/EF4+pQS6Xds4Z7KOE2ywggFVEfG0bcdeT1JJOSeGrrVLrxVpFvbzrNPLewpHHeOzQOxcACQDkoTwR6Zo/e+Qf7D/AH/wPtqz1Cy1GEzWN3BdRK20vBIHUHrjIPXkfnViuP8Ah/FHBD4ihhjWOKPXLlURBhVA2gAAdBXYVVOTlFNmWLoxo1pU4u6QVDdXVvZW73F3cRQQJjdJK4RVycDJPHUgVNXJfEz/AJJ7qn/bL/0alFSXLBy7CwlFV8RTot25ml97sa//AAlPh7/oPaX/AOBkf+NH/CU+Hv8AoPaX/wCBkf8AjXz/AP8AC8fD3/RM9L/7/R//ABmug8L/ABY8Da9qNnpt54Hhs768u0toRDbwTR/OVVWZiEI5JyAp4HfpU/vfI2/2H+/+B7B/wlPh7/oPaX/4GR/40f8ACU+Hv+g9pf8A4GR/40f8It4e/wCgDpf/AIBx/wCFU9V0vwfoel3Gp6npWkW1nbpvlle0jwo/75ySTgADkkgDJNH73yD/AGH+/wDgXP8AhKfD3/Qe0v8A8DI/8aP+Ep8Pf9B7S/8AwMj/AMa8DtvjT4YRNUN14E0+ZxcE6cIraKIPCW4ExO7a4XnK7gScYXGToeG/i54Y1zxLpukz/DzT7VL24S3EyGKUoznap2mJcjcRnngZPPQn73yD/Yf7/wCB7Z/wlPh7/oPaX/4GR/40f8JT4e/6D2l/+Bkf+NH/AAi3h7/oA6X/AOAcf+Fcvo+ufDDX9cfRdLTRLm/XfiNbEAPt+9sYoFfuflJyASOBmj975B/sP9/8DqP+Ep8Pf9B7S/8AwMj/AMaP+Ep8Pf8AQe0v/wADI/8AGj/hFvD3/QB0v/wDj/wo/wCEW8Pf9AHS/wDwDj/wo/e+Qf7D/f8AwD/hKfD3/Qe0v/wMj/xo/wCEp8Pf9B7S/wDwMj/xo/4Rbw9/0AdL/wDAOP8Awo/4Rbw9/wBAHS//AADj/wAKP3vkH+w/3/wD/hKfD3/Qe0v/AMDI/wDGj/hKfD3/AEHtL/8AAyP/ABryv42avaeCdL0y30XQdIhu9ReQ/amsYnMSx7MhVZSCW3gZOcAHjJBHz/qXiTVtVuFnuLlUdUCAWsKW64yTysaqCeeuM9PQUfvfIP8AYf7/AOB9qf8ACU+Hv+g9pf8A4GR/40f8JT4e/wCg9pf/AIGR/wCNeX/Aeyste8A3EurWFlezQahJDHLPbI77NkbYLEZbl26k+nQAV6h/wi3h7/oA6X/4Bx/4UfvfIP8AYf7/AOAf8JT4e/6D2l/+Bkf+NH/CU+Hv+g9pf/gZH/jR/wAIt4e/6AOl/wDgHH/hR/wi3h7/AKAOl/8AgHH/AIUfvfIP9h/v/gH/AAlPh7/oPaX/AOBkf+NH/CU+Hv8AoPaX/wCBkf8AjR/wi3h7/oA6X/4Bx/4Uf8It4e/6AOl/+Acf+FH73yD/AGH+/wDgH/CU+Hv+g9pf/gZH/jViz1rStRmMNjqdndSqu4pBOrsB0zgHpyPzr5D+I3iG9ufGurWUKW1jaWF7PbwQ2MCwAKrbMsVALE7QfmJwScYBxXU/A/VL5PGWmxi5dlu554Jt/wAxaMQNIFyeR8yKePT3NJyqRavY0hRwtWM/ZuV0m9bW0PqGiiitjzQooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA5Lx3/wAy1/2HrX/2auM/aO/5J5p//YVj/wDRUtdn47/5lr/sPWv/ALNXGftHf8k80/8A7Csf/oqWsofHL5HdX/3Wj/29+Z8yTQTWzhJ4pInKK4V1KkqyhlPPYqQQe4INfS/7PXiuHUfCcvhuUxpd6W7PEowDJDIxbPXJKuWBIAADJ3NeYat4GvtT+DvhnxXp0Uk4sre5gvY1OSkQuZWWQLjJALPuOeBtOMBiM/4Oa+3h/wCJulNuk8i+f7DMqKrFhIQEHPQCTyySOcA9eh1OE+w6+YPjr4rt/Fnii30TSbaeZtE+0rPKFPzPgGQBcZ2oIjlj/tHoAT7v8RtZ/sD4da9qIeeORbRoopIDh0kk/dowORjDOpyORjjmvL/2fPBe3R9T8Q6naQS2+pxGztg7b98IZhMGT7u1mVRzz8h4APIB88V9/wBfAFff9AHB+Lfi94U8HX8unXs13cahC6rLa2sBLIGTeGLNtQjBHRieRx1xJ8O/EPgrX/7W/wCEQ02Cy+zSrHceVZLb+cvzeW/yjlTh8Zww5yBnnz/9ozwxpcWk2XieKDytTku0tJnTgTIY3ILDuw2AA+hwc4XGZ+zTpsMus6/qjNJ59vbxW6KCNpWRmZieM5zEuOe569gD6Lr4wsNJTTPjJbaNZ3M8cdr4gS1hn+UyKFuAqtyu0sMA8rjPbHFfZ9fIkMay/tElXmjhA8UM25wxBIuSQvygnJIwO2SMkDJAB9d1z/iTxx4a8I+WNc1eC0kkwVhw0khBzhtiAtt+UjdjGRjOaseKvENv4U8L6jrl0u+O0iLhMkeY5OETIBxuYqM44zk8V8qeCvDt98WfiDMurapJvdGvL64bmR0Uqu1BjAPzKB2UdAcBSAe56N8fPBWr6ilnK19pu/AWa+iVYyxIABZGbb1zlsKADkivMP2gfEFnrfiHSF03WrHULGC0YiO0kEnlSM53Esox8wVONxI2k4GQW9rj+EvgOLS5tOXw1aGCV97OzO0oPH3ZS29R8o4DAdfU5+YPiN4MbwL4xuNIWSSW0ZFntJZNu54mzjOD1DBl6DO3OACKAPR/hX/yTzSf+xzh/wDRSV9GV85/Cv8A5J5pP/Y5w/8AopK+jKyh8cvkd1f/AHWj/wBvfmFFFFanCedfGexuNT8G21hZx+ZdXV8sMKbgNztHIFGTwMkjrXyroWp/2J4h0zVvJ877DdxXPlbtu/Y4bbnBxnGM4NfYXjv/AJlr/sPWv/s1fPvxG+DOqeDv9P0oz6rpB3s7rF+8tQMn94B1UKP9YMDIOQvGcofHL5HdX/3Wj/29+Z9R6Vqtjrml2+p6Zcx3NncJvilTow/mCDkEHkEEHBFZ/jDw1D4w8J3+gz3Elul2igTIASjKwdTg9RuUZHGRnkda+JLG/vNMvI7ywu57S6jzsmgkMbrkEHDDkZBI/GvcPAXx+1FtRtNK8VxQTwzyrF/aSlYWi3E/NIOEKglRkbcKCfmNanCez+CfDH/CHeELHQPtn2z7L5n7/wAry926Rn+7k4xux17Uf8J34P8A+hr0P/wYw/8AxVamrabDrOjX2l3DSLBe28lvI0ZAYK6lSRkEZwfQ18cfEPwPN4A8SrpMt9HepJbpcRTLGUJUllwy5ODuVuhPGD3wAD7H1XVbHQ9LuNT1O5jtrO3TfLK/RR/MknAAHJJAGSa5Ox+MHgHULyO1h8RwJI+cGeKSFBgE8u6hR07nnp1rwjwB8LfEHxE0dJbzVZ7DQrTetk0ytMrOzZcRRlgAuQdzAjLDHJB29Rrn7Ov9m+Er27sNYnv9Xt90yRC22JNGq5MaoCzeYecHJB4XAzuAB9BwTw3VvFcW8sc0EqB45I2DK6kZBBHBBHOakr5U+Bnjabw94xh0W5uJP7L1V/K8sklY7g4EbgAE5JAQ4wPmBJ+UV9V0AfDmjaZrVr4v0e1gjk0/VJL2JbR7qIqElE2wMQVPCyKQeDypGOCK+46+JILC+vPiXFpziPRtQm1gQN9jXC2Upm2/uwrcBGPGG7DB719l63reneHNHn1bVrj7PYwbfMl2M+3cwUcKCTyQOBQBJqWrabo1utxqmoWljAzhFkupliUtgnALEDOATj2Nc3/wtTwN/bH9l/8ACTWP2j+/uPk/d3f67Hl9P9rrx14r58M/iP47/EFLdpY7W2iRnSMtujsrcEBiBwXckqCeCxI+6o+XsJ/2Zplt5Wt/Fcck4QmNJLAorNjgFhISBnvg49DQB73Y39nqdnHeWF3Bd2smdk0EgkRsEg4YcHBBH4UX1/Z6ZZyXl/dwWlrHjfNPII0XJAGWPAySB+NfLHws8Wa74G8fR+GZk32t3qC2N5ZO4IimLiMyKRkBgeuOGAx/dI6z9ofwlq91qVv4qghjk0u2skt7hxIA0Tea2CVOCQTKoG3PQ5xxkA9303VtN1m3a40vULS+gVyjSWsyyqGwDglSRnBBx7iq+t+JNF8OW/n6zqlpYoUd0E0oVpAoy2xerkZHCgnketfEmh32qadrlldaJJPHqaSqLYwLucueAoXndnONuDnOMHNeyR/s/eJtcsptU17xJGutSpuEUwa4LEINqyTFsgg/KcBwABgnpQB9B2N/Z6nZx3lhdwXdrJnZNBIJEbBIOGHBwQR+FWK+PNI1vxH8GvHl3aP5bvC6x3toJMxXMeNykHsdrblbGVzgjllP1/BPDdW8VxbyxzQSoHjkjYMrqRkEEcEEc5oAkrLm8S6DbaoNLn1vTYtQLqgtHukWUs2No2E5ycjAxzkV82fFP4w3niq8k0rQLme00KPcjOjGN73IKkt3EZBICHrnLc4C9Po37NyS6ck2r67PFdS2gP2eG3UfZ5yAcM25hIqnIIG3d2IoA9A+Nv8AySHXf+3f/wBKI68k/Zvgmbx1qdwsUhgTTGR5Ap2qzSxlQT0BIViB32n0qP4pfD/xB4H0dY9K1jVb3we21XtpLhits5bOJEGFKl+QwUfMcHnBbY/ZohuG1HxFMt1ttUigSS38sHzHJcq+7qNoVxjvv9hQB9D1T1LVtN0a3W41TULSxgZwiyXUyxKWwTgFiBnAJx7Guf8AiN4zXwL4OuNXWOOW7Z1gtIpN215WzjOB0Chm6jO3GQSK+ePA/gTWvi54gvNa1bUZBaLcL9uu3B8yUkE7Ivl2ZACjHAQMuARgUAfT+m+JdB1m4a30vW9Nvp1Qu0drdJKwXIGSFJOMkDPuK1K+aPG3wBuPD+h6hrOkax9uhtd0zWs0IjdYBkk792GZRgnhc4OOcKd/4KfFXVNa1geF/EN19qkkiLWN04/eEoozGxA+b5Qzbm5ypyWLDAB7xRXzB8VvA2qWXjltX8R6xOfDt9d+XDqWPPa0Dl3WHyi4bamG+7xt5HJ21X/Z5huJPiVI8F15McWnyvOnlhvOTcgCZP3fmKtkf3MdCaAPqeivN/jD4d8Y+JNDtrbwrd7IV8wXtolx5L3QbaqqCcAqAZMqzAH0JAr5k8PT+I9G8VQW+iS3djrTXC2qxq3lMZN4HluGwMbgAVbjjmgD7jorz/4wWXim88Gxf8Ik98t9Ddiab7DOYpDCI5NwGCC3JX5Rkk4wDXyZbatqVlcW1xaahdwT2qFLeSKZlaFSWJCEHKgl3JA/vH1NAH3fNPDbIHnljiQuqBnYKCzMFUc9yxAA7kgVy3h//koXjH/ty/8ARRr5R8W+LfEfjN7TUddmkliiQ28BWPZFuVV8wgDjecqzY/vL0G0D2v8AZ8Sa8t72/nvbt3ht47cRtKSjLuYKSD1KLGFX0BYd6yqfFH1/RndhP4Nf/Av/AEuB7hRRRWpwmT4p/wCRQ1r/AK8J/wD0W1Hhb/kUNF/68IP/AEWtHin/AJFDWv8Arwn/APRbV4P48+LGsaDpuneF9CnghU6Lai5uVjcTwSMgYhGzgZQpyASNxwQRxl/y9+R3f8wP/b/6H0FqWrabo1utxqmoWljAzhFkupliUtgnALEDOATj2NSWt/Z33n/Y7uC48iVoJvJkD+XIv3kbHRhkZB5FfMGm/BXxx4s0PS9Wm1Sx8mW0QWsd7dyu8UHWNRhGCrg5Cg8Z6A5FYeteFfG/wk1SHURNJahnEcV/YylopSNr7G6HGR911AbaeCBWpwn2HXz/APtKQaiP7DuN/maY29Nn2Vf3Mw5z5uNw3qfuZAPlZ5xx6Z8LfGbeOPBUGoTxyLeW7/ZLtm24klVVJdcADDBgcYGCSOQAT88fFX4k6l421Q6fLZyafp+n3EgitZNyyluFzMucbxhsAD5dzDJ6kA7j9maCFrjxLcNFGZ0S2RJCo3KrGUsAeoBKqSO+0elereO/+Za/7D1r/wCzV8+/Dj4wf8K/8PXGk/2F9v8AOu2ufN+1+VjKIu3Gxv7mc5713+gfEPUviBpejzajpUdq9n4gsk+0wFvKnZvMJChs7SqhMjc33geMgVlW+Bndlv8AvUfn+TPcKKKK1OEyfFP/ACKGtf8AXhP/AOi2ryv4sabNffAPRriJowlgtncShiclTH5WF467pFPOOAfofVPFP/Ioa1/14T/+i2r5P8c+JfGN/ptlpetwz22iIUOnxPZ+UkgjTYHVyNz5Vsn5iPn6AYAy/wCXvyO7/mB/7f8A0Nv9n2ya6+JomWWNBaWUszK0KuXBKphSeUOXB3DnAK9GNfVdfFnw5k8Vw+MbeXwdDJPqaIzNECBG8QxuWTJA2Hgckc7cENtr6f8AA+peP7+4vF8ZaFpunQIim3e1myzNk5BUO4IxjklcccNk7dThPnT42/8AJXtd/wC3f/0njr6f8Cf8k88Nf9gq1/8ARS18wfG3/kr2u/8Abv8A+k8ddp4d8Q/Evxv4DstE8M6baWmn2yRadLqsV0YpY2jKncDvDKNmwNhWz82Ou0AH0XRXyJrWm+P/AIa+JYfEWrXUn2t7gIty+o+Yb5YypwwDiRoiFTIYDgqDgkCvo/4c+M18deDrfV2jjiu1doLuKPdtSVcZxkdCpVupxuxkkGgDrKK5fx1460vwHoZv78+bcSZW1tEbD3DjsPRRkZbtnuSAfnyP4i/FTxxqk39iPd4dPszQ6ZbYigEmBkuclCShIdmyvzYKjNAH1XXJeO/+Za/7D1r/AOzV8y3niv4k+DtcEWo61rlrfQ7iIr2d5UYcruCuWR1yDhsEcZB4zXq+kfE6x8caX4TsJ5JF8QW+rWrXcTJxIF+UyqwAXDFh8vBBJGCACcq3wM7st/3qPz/JnuFFFFanCFcl8M/+Se6X/wBtf/Rr11tfI/iXxd430jSdL0+C9u9M0VkaSyktGMRnId95MincSGYgrkDAQ7eQTk/4q9H+h3U/9xqf44flUPriiviD/hO/GH/Q165/4MZv/iq+z9C/tH/hHtM/tf8A5Cf2SL7Z93/XbBv+78v3s9OPStThNCivkj4g+NvHsHjK/jvNS1zSIxLJ9ktfMNtiDzH2cRkK/cbwWzj7xAFex/A3X/FfiDw1dz+IWkns43SOwvJlAeYAFXBPVwCq/MRkktliRgAHqlFeB+NPjdq+oay3h/wBZySzq7xtdJCLiSVkY8wKpZSm1SdxByG6LjJ5S6+Jnxf0HyL3V/t0FqJVGL7SUijlPXYW8tTyAehBxnBFAH1PRXH/AA/+Iml/EHTp5rGGe2urTYLq3mGdhYEgqw4ZchgDwfl5AyK4/wCLXxa1jwJ4hstJ0mwsZfNtBcyS3Yd87nZQoCsuMbCcknOR0xyAewUV8qaB8fPGGm6os+sTx6xZ7CrWzRxwHPZldEyCD6gjBPGcEV9S+PPjy+uFlt76009AgUxWtojKTk/MfM3nPOOuOBx1yAfU+rWc2o6NfWVvdyWc9xbyRR3Med0LMpAcYIOQTnqOnUV8eWPwt8a3uuR6SfD19byNKYmuJ4WWBMZyxlxtKjBOQTntkkZ9buvj9qUHg6x1NfB9359wjxPeShksvOGQvltgmQEqxK5UjaRk/erI8N/tCeIb7xLptnqel6a1ncXCQy/ZIJvNAY7cqNzliCQdoUlsYGCc0Ae96Fpn9ieHtM0nzvO+w2kVt5u3bv2IF3YycZxnGTWhRXh/xO+ONxoGsXvh7w5bQPdW/wC7m1CVhIqOVO5UQcblJXljwyspU9aAPcKK+WNG+Jnxf1PZe6d9u1W1ilAcQ6SkkbEYJRmjjyMgjOCDg9RXT+E/2jHl1FofFthBDaybRHcafE37o5wS6sxJXBzleRt6NngA+gKK83+LvxHvPh/p2mf2bbQTX19K+PtKFoxGgG77rKd2XTHUY3e1eWT/ALSHipriVrfStGjgLkxpJHK7KueAWDgE474GfQUAfTdFcn4g8cQ6N8N/+Eyt7GS5ga3t7iO2kkETFZWQAEgMAQHz36fjXjmp/tKaxL5X9k+H7G1xnzPtcz3G7pjG3Zjv1znI6Y5APo+iuH+FXje88e+EpNTv7WC3uobt7ZxATsfCqwYAkkcOBjJ6Z74HH/GD4wXHhi8bw54cby9WTY91dvGGEAIDBFDAhmIIJJBABwMk/KAe0UV8gf8AC7fiH/0MP/klb/8Axuuo+GnxV8bav8RbCxvbr+1Yb/EEsDiOIRou5jKu0ABlG4nj5gNvUKVAPpeiiigAoryP4ufFPXvAWs6fZaXpdpJBcW5la5u0dldtxGxdpUZUAE8n768Dv55/w0d4w/6Buh/9+Jv/AI7QB9P0Vz/gfXrzxP4L0vWb+y+xXV3FveEAgcEgMuedrABh14YcnqfENc/aI8SW2uXtvYaJY21rFK0aQ38MhnXbwfMw4AbIOVxx0ycZIB9H0V5f8IvibqnxA/tOHU9Mghks9ji4tPljw2QEKsxbd8rHIyMZztIG7tPFHi7RfB2lvf6xexwgIzRQBgZZyMDbGmcscsvsM5JA5oA3KK+f/wDhpr/qUf8Aypf/AGqvRPA3xZ8OeOHjs4HkstWZCxsrgcthQW2OOHAyfRsKTtAFAHeUVXv7630zTrm/vJPLtbWJ5pn2k7UUEscDk4APSvnTUv2k9eluFbS9D022g2AMl0zzsWyeQylABjHGOx554APpOivmD/ho7xh/0DdD/wC/E3/x2vZ9d+Jul+HfAOneJb+PbcalaRz2unpJl5HdA20Nj7q7hl8cemSFIB4xc/tGeLH1Fpraw0qK1G8JbvE78EgqWbcCWUDGRtB3HI6Y+l7Ca4uNOtpry1+yXUkSPNb+YJPKcgFk3DhsHIyOuK+CK+p7H9obwVd3kcE0Wq2UbZzPPbqUTAJ5COzc9OAevpzQB6xRUc88Nrby3FxLHDBEheSSRgqooGSSTwABzmvD/EP7SFjbXE9v4f0WS8QIyx3d1L5Sl8kBhGASyfdPJUnJGF60Ae6UV4v4e/aM0LULwwa5pk+kRn7k6SG5QcEncAoYdABgNnPOAM17BY39nqdnHeWF3Bd2smdk0EgkRsEg4YcHBBH4UAWKKx/EninRvCOnR3+uXn2S1klEKv5TyZcgkDCAnop/KvFNV/aWmZLiPR/DcaPvxBPd3JYbd3Vo1A5K9g/BPU45APQ/jb/ySHXf+3f/ANKI6+YPAn/JQ/DX/YVtf/Rq165rnx70XxJ4O1bSbzRtStLi8snijNvOGUSneAGYFDs4jJ4OdzKVIHzeV/DixuNQ+JXhyG1j8yRdQhmI3AYSNhI559FVj7445oA+qvAn/My/9h66/wDZa62uS8Cf8zL/ANh66/8AZa62sqPwI7sy/wB6l8vyQVyXxM/5J7qn/bL/ANGpXW1yXxM/5J7qn/bL/wBGpRX/AIUvRhlf+/Uf8cfzR5L8Y/hDpei6PeeK9Cf7LHHKrXVif9WA7KgMWB8vzHJU8YY42hQp8Hr6/wDjb/ySHXf+3f8A9KI68Q0T4e2/i/4PS63o1hONf0u7kilSKQyfb0+R/uE/Kyq/AXOdhGCWGNThO7+AvxDW6sh4R1e8ke8jfGm79zF4ghZo8hcAIEJBZujBRwoFe0arpVjrml3GmanbR3NncJslifow/mCDggjkEAjBFfClhfXGmajbX9nJ5d1aypNC+0Ha6kFTg8HBA619x6B4j0jxTpa6lot9Hd2hcpvUFSrDqGVgCp6HBA4IPQigD58+NHwt0LwfpNtrmhtPbxz3a2z2buZEXMZIZWPzD7hyCWzu4wBg9P8AA/4b6E/hzT/F99D9s1KWV5LYSZCW2x2QYXOGbI3bj0+XABGToftHf8k80/8A7Csf/oqWug+CX/JIdC/7eP8A0okoAPjWlxL8KNYitrOe5ZvKL+SAfKRZFdnYZztAXnAOM5IwCR8ueENE1LX/ABVp1lpf2uOc3EZa5tY2drVd6jzjtwQFJBzkY45FfR/jP45aL4T8Spo8FnJqZifbfywTBRbnIyq5BDuBnIyoBwM5zty/AHxr0fW/FCaL/wAI1Bon9pSvJ58VwhWW4Iz842JlnxjdyS20Y5yAD2iiivI/Gfx70Xw5qiWGj2keukJumnguwkSE4IVXCsHOOuOBwMk5AAPXKK+ZIP2kPFS3ETXGlaNJAHBkSOOVGZc8gMXIBx3wcehqTWP2jNdvtDS103TINN1I7PMvVkEo4+9sjZcLk/3i2BkcnBAB9H31hZ6nZyWd/aQXdrJjfDPGJEbBBGVPBwQD+FfDniXTYdG8Vavpdu0jQWV7NbxtIQWKo5UE4AGcD0FfR/wN8d6/4zt9bi166juns3haKUQrG2HD5U7QAQNgI4zyeTxjxz42/wDJXtd/7d//AEnjoA9z+A2mw2Pwrs7iJpC9/cTXEoYjAYOYsLx02xqec8k/QemV4H8P/i94U8HfDDSNOvZru41CF5VltbWAlkDSyOGLNtQjBHRieRx1xY039pbTZbhl1Tw3d20GwlXtblZ2LZHBVggAxnnPYcc8AHulcX8T/HE3gHwqmqW9jHdzz3AtY1kkKqjMjsHOBlgCn3cjOeorqNN1bTdZt2uNL1C0voFco0lrMsqhsA4JUkZwQce4rwT9o+/0K4vNMs47ueTXbXPmQpITDFCwz8y9FkJ2kY5K/e42UAa/wz+NeseLvGkGh6zZWMcd1FJ5D2cTgiRRv+YtIfl2q/QZzt7Zr3Cviz4aa/Y+F/iHpGsam0i2cDyLK6LuKB42TdjqQCwJxk4BwCeK+q/D3xF8I+Krw2ej63BPdDpC6vE78E/KrgFsBSTtzjvigD5U+J2mzaV8TfEVvO0bO969wChJG2U+ao5A52uAffPXrXZ/AbTZr7xZZ3ETRhLCWa4lDE5KmExYXjrukU844B+hs/tI2NxH4y0m/aPFrNp/kxvuHzOkjlhjrwJE/P2NW/2cf+Q3qH/XtJ/6FFWVXp6ndgf+Xn+B/ofRlFFFanCFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAHJeO/wDmWv8AsPWv/s1cZ+0d/wAk80//ALCsf/oqWuv+IU8Nrb+Hri4ljhgi1u2eSSRgqooDkkk8AAc5rw342fEDw541fTLfRftc72DyH7Uy+XE6uqZCqw3k5UDJ24weGyCMofHL5HdX/wB1o/8Ab35nq/wo02HWfgRYaXcNIsF7b3dvI0ZAYK80qkjIIzg+hr5c1zRrzw9rl7pF+my6tJWifAIDY6MuQCVIwQccgg17/wDA/wCJGhJ4c0/whfTfY9SileO2MmSlzvdnGGxhWydu09flwSTgZH7R/hhILzTPE9tBt+0ZtLtxtALgZjJH3ixUOCeeEUccZ1OEk8ffEi08efD7R/D+iX0c3iDVLi1jvbKOB41LEZKB5BtAE2zkN26kZr3uwsbfTNOtrCzj8u1tYkhhTcTtRQAoyeTgAda+aP2efDf9p+NLjXHk2x6RF8qBsFpJQyDIxyoUSZ5Bzt6jNe3/ABE8V6X4f8JazFNrUFlqb6fKbWJZ9s5dlZY2RQd33/4gOME5GCQAfGFff9fAFfeelarY65pdvqemXMdzZ3Cb4pU6MP5gg5BB5BBBwRQB5X+0d/yTzT/+wrH/AOipa5/9mX/maf8At0/9rUftD+MNLvdOsPDVjcwXV1Fdm4ujDLu+zlA0YRsDG4lmyM5XZyPmFUP2eNd0fRP+Ek/tbVbGw877N5f2u4SLfjzc43EZxkdPUUAfR9fGmk6lDrPxssdUt1kWC98Rx3EayABgr3IYA4JGcH1NfYd9f2emWcl5f3cFpax43zTyCNFyQBljwMkgfjXxR4Tv/wDi4eh6jqN3/wAxW3nuLm4k/wCmqszux/EkmgD3/wDaO/5J5p//AGFY/wD0VLXOfszQQtceJbhoozOiWyJIVG5VYylgD1AJVSR32j0r0/4qeEpvGfgO806zhjl1CJ0uLMPIUHmKeRnpkoXUbuMsM46j58+DPjpPBvi0296caZqmyCdiyqInDfJKxb+FdzA8jhiedoFAH1vXzB+0d/yUPT/+wVH/AOjZa+l7G/s9Ts47ywu4Lu1kzsmgkEiNgkHDDg4II/Cvlj44+NLPxZ4tt7bSrv7TpumxGJXVRsaZm/eMjdWXAQZ6fKSODkgHR/Cv/knmk/8AY5w/+ikr6Mr57+Hms6PcfD/wnpVs8EWp23iONrqHKLJLlmIl2g7mXa6JuI6rjsK+hKyh8cvkd1f/AHWj/wBvfmFFFFanCcV8Sb630zTtDv7yTy7W11mCaZ9pO1FDljgcnAB6Vq+DvGOl+OND/tbSfPWFZWhkjnTa8bjBwcEg8FTwT19cgcl8df8Aknn/AG8/+0pa8x+AXjltH8QHwveSxrp+pOXgLBRsucAD5iRwyrtxyS2wDGTnKHxy+R3V/wDdaP8A29+Z7H4n+Eng7xVLNc3WmfZb6blruybynJ3bixHKMxJOWZSTnrwMfNnj74Z614AuI2vDHdafO7LBewg7SQThXB+45UZxyOuCcHH2XXg/7RnifS5dJsvDEU/m6nHdpdzInIhQRuAGPZjvBA9Bk4yudThNz9nzxHNq/gq6027vpLi40y4CRo4JMVuyjyxuxyNyyADJIAA4GK4j9pPTZovFWjaozR+RcWRt0UE7g0blmJ4xjEq457Hp37/4FeD77wt4VvZ9W02Sy1C/uFfEj/M0ARTHlQTtIZpOCA3PPQVxn7S9jbx6j4dv1jxdTRTwyPuPzIhQqMdODI/5+woA9j+H0ENt8OfDaQRRxIdMt3KooUFmjDMeO5Ykk9ySa6SuT+Geq2OrfDnQXsLmOcW9lDazbescqRqrIw6gg/mCCMgg11E88Nrby3FxLHDBEheSSRgqooGSSTwABzmgD4s+I9r9j+JXiOL7RBPu1CaXfA+5RvYvtJ/vLu2sOzAjtX2vXxho+l3HxJ+JziGynWHUtQe6ukhcE28DybpG3kY+UNgEjk4GMkCvs+gD48gghtfj7Fb28UcMEXigJHHGoVUUXWAABwABxivY/wBo7/knmn/9hWP/ANFS14B4E/5KH4a/7Ctr/wCjVr6D/aH027vPh9b3cDSGCyvUkuIwUC7WDIHORuJDMqgKf4ySDgEAHinwy0/xs+uSav4KtvPuLDaLhWmjRHR8/I4Zl3K2w9OmAQQQDXq+p6j8fb/yvs2jWOm7M7vsj2zeZnGM+bI/THbHU5zxjnP2ctfsdP8AEGqaLcNIt3qiRta4XKsYhIzKT2O1sjt8p5zgH6ToA+UP+FMfErXtY83VrbbNN/rL6/v0l6LxuKs7ngBRgHt0FfR/jv8A5J54l/7BV1/6KauT+L3j3WvBqaHZ6BaRz3+p3BClkMhIRk/dqg5JcuFyDkDOOSCOs8d/8k88S/8AYKuv/RTUAfLnwYghufi3oKTxRyoHlcK6hgGWF2U89wwBB7EA19h18mfAaOxf4qWbXc0kc6W8zWaqOJJdhBVuDx5ZkPblRz2P1nQB8gfG3/kr2u/9u/8A6Tx19L3/APbv/CrLnz/P/wCEh/sR9/2fHmfavIOdnl/xb+m3v0r5c+LUd9F8VPEC6jNHNObgMrIMARFFMS9ByIygPuDyep+r4DD4z8CxNcRyW0GtaYDIkbgtGs0XIDEYJAbrjt0oA+RPhxD5/wASvDifZp7jGoQvsgOGG1g248H5Vxubj7qnkdR9r18OQGbwZ46ia4jjuZ9F1MGRI3IWRoZeQGIyASvXHfpX2vpWq2OuaXb6nplzHc2dwm+KVOjD+YIOQQeQQQcEUAcX8bf+SQ67/wBu/wD6UR1wf7NcE1s/idJ4pInKWThXUqSrLKynnsVIIPcEGu0+Ot9b2nwo1KGeTZJdywQwDaTvcSLIRx0+VGPPp64rhP2Zp4VuPEtu0sYndLZ0jLDcyqZQxA6kAsoJ7bh60AY/7RWvrqHjGx0WJo2TS7ctJhWDLLLhipJ4I2LERj+8eew9z+HFjb6f8NfDkNrH5cbafDMRuJy8iiRzz6szH2zxxXjH7SOjXieIdJ1zZusZbT7HvUE7JEd3wxxgZD8c5O1uOK9b+E+tw658MtDmi8tXtrdbOWNZA5Rohs+b0LKFfB6Bh16kA7SvkTwhDY+Gfj1a2DG7e0tNYlsYmSTEhO5ooyxUrkbiu4dCNwwQcH6zv7630zTrm/vJPLtbWJ5pn2k7UUEscDk4APSvkzwDBN4l+OGmXGrRSW093evqjJGpT5trXKYDZOwkKfdTwec0Ae9/G3/kkOu/9u//AKUR15B+zj/yUPUP+wVJ/wCjYq9f+Nv/ACSHXf8At3/9KI68g/Zx/wCSh6h/2CpP/RsVAH0/XxR4TvrjU/inod/eSeZdXWt280z7QNztOpY4HAySelfa9fGHwr/s7/haHh7+1P8Aj3+1jZ97/XYPk/d5/wBZs9vXjNAH2fXwhrv9o/8ACQ6n/a//ACE/tcv2z7v+u3nf935fvZ6celfd9fEHjv8A5KH4l/7Ct1/6NagD6z8L+GtNm+HPh3S9V0S0dIbKB3tLq1UhJvL+clGHD7mfJxnJOepql4K0qx0Pxd4o0zTLaO2s7dLJIok6KPKP4kk5JJ5JJJyTXawCZbeJbiSOScIBI8aFFZsckKSSBntk49TXLeH/APkoXjH/ALcv/RRrKp8UfX9Gd2E/g1/8C/8AS4HW0UUVqcJk+Kf+RQ1r/rwn/wDRbV8V3MN9rnig2cRkuby4uVtYFeTljkJGmWOAANqjJwAB0Ar7U8U/8ihrX/XhP/6LaviuLUptG8WpqlusbT2V8LiNZASpZJNwBwQcZHqKy/5e/I7v+YH/ALf/AEPt7SdNh0bRrHS7dpGgsreO3jaQgsVRQoJwAM4HoK5f4saJDrnwy1yGXy1e2t2vIpGjDlGiG/5fQsoZMjoGPXoessL631PTra/s5PMtbqJJoX2kbkYAqcHkZBHWuT+LGtw6H8Mtcml8tnubdrOKNpAhdpRs+X1KqWfA6hT06jU4Tyj9mzUtSGs6zparI+lm3Fw7EMVimDBVA52gspbPGT5Y/u1b/aU0zTl/sPVvO2anJvtvK2sfNhX5t2c7V2M2MYyfM/2aP2Zf+Zp/7dP/AGtR+01/zK3/AG9/+0aAOg/Zx/5J5qH/AGFZP/RUVdn47/5lr/sPWv8A7NXGfs4/8k81D/sKyf8AoqKuz8d/8y1/2HrX/wBmrKt8DO7Lf96j8/yZ1tFFFanCZPin/kUNa/68J/8A0W1eN/GaeFfgt4St2ljE7yWzpGWG5lW3YMQOpALKCe24eteyeKf+RQ1r/rwn/wDRbV4/8Zf+SHeFf+u1p/6TSVl/y9+R3f8AMD/2/wDoZX7NMNi2s6/PIY/7QS3iSAGTDeUzMZMLnkbliycccdM8/RdfP/7Mv/M0/wDbp/7Wr6ArU4T5E+OJhPxb1cRRyK4SASlnDBm8lOVGBtG3aMHPIJzzgfT/AIO0BfC3g7StFVYw9rbqsvlszK0p+aRgW5wXLHt16DpXy58bf+Sva7/27/8ApPHX1vYfY/7Otv7O8j7D5SfZ/s+PL8vA27McbcYxjjFAHmf7QVnDc/DIzS3ccD2t7FLFG2MzsQybF5HO12fjPCHjuOT/AGZf+Zp/7dP/AGtW5+0hPCvgXTLdpYxO+pq6RlhuZVikDEDqQCygntuHrWP+zNBMtv4luGikEDvbIkhU7WZRKWAPQkBlJHbcPWgDkPjnrp134kTae0sltbaVb+SguPMCvJtMjMqbcgsSqA4w21W3bTkaHwm+LOg+A/Ct1peqWmpTTy3r3CtaxoyhSiLg7nU5yh7elcn8XtG/sT4oa1EqTiG5lF5G8w+/5oDsVOBlQ5dR/u4ySDX0f4T8IeE9Q8G6Hey+E9DElxp9vKw+wo+C0ak/M4LHr1YknuSaAPLPiz8S/BfjPwc9rYQSXGqQXvl2sk0TRtHH1aZDggowAXYxB5BK/KK8++Ff/JQ9J/6+Yf8A0alfUEPwx8EQaWdOTwxppgKMm94Q8uGzn962Xzzwd2RxjGBXl9tqfw6i8W6Z4f8ABml7Lq31mCWTUFHmJKoYKypKzFyu4rx907SRngnKt8DO7Lf96j8/yZ75RRRWpwhXDeCdNh1n4SQ6XcNIsF7b3NvI0ZAYK7yKSMgjOD6Gu5rkvhn/AMk90v8A7a/+jXrJ/wAVej/Q7qf+41P8cPyqHx7rumf2J4h1PSfO877Ddy23m7du/Y5XdjJxnGcZNfa/hO+uNT8G6Hf3knmXV1p9vNM+0Dc7RqWOBwMknpXhH7RPhH7HrFp4rtl/c32La756TKvyNy38SLjAAA8vJ5auT8LfFGbwh8OdS8O6daSHUL64lf7Z5pQW6tHGgKbeS/yvg5G07T83IrU4TcvIX+Knx+Flc3UF9pNvKyCWyjYxC0iJYKWGD8xO0vnG5/lJG0V9L38Nxcadcw2d19kupInSG48sSeU5BCvtPDYODg9cV538D/CUPh3wHb6i8Miahq6LcTlpAwMeW8kKBwBsbd65c56ADuPEev2Phbw/ea1qTSC0tUDP5a7mYkhVUD1LEDnA55IHNAHiHwb+G3ivw743k1TVtKtLW3tka3f7UQ8hLpuDQFcjI+VS2cYZ1GTkDv8A44zww/CTV0lljR5ngSJWYAu3nI2F9TtVjgdgT2ryDR/FXxD+LnjGHTrTXZNJiRDNL9gkaCOCIbVZsBt8hzjALHluqjJHX+N/gp4Z0jwdrutHV9SOoRobsXeoXCvvcbiUPC5MjMBk5O7bjuGAOc/ZuvriPxlq1gsmLWbT/OkTaPmdJECnPXgSP+fsK+i9W02HWdGvtLuGkWC9t5LeRoyAwV1KkjIIzg+hr54/ZsnhXxVrNu0t2J3sg6Rqw8hlVwGLjqXBZQp7Bn9a+g9duryx8Paneadb/aL6C0llt4dhfzJFQlV2jk5IAwOTQB8SeGtNh1nxVpGl3DSLBe3sNvI0ZAYK7hSRkEZwfQ19V6n8FvA2o6PFp0ekfYfJz5dzaSETDLBjlm3b+mPn3YBOMV88fB+xt9Q+K+gQ3UfmRrK8wG4jDxxvIh49GVT7454r7HoAz/7E07/hHv7A+z/8Sz7J9i8je3+p2bNu7O77vGc596+RPhLBaXPxU8PpexSSxC4LqqK7ESKjNGfl5wHCknoACW4zX2HfzXFvp1zNZ2v2u6jid4bfzBH5rgEqm48Lk4GT0zXxx8K7y3sfih4eluhOY2uxEPIcq29wUTJBHy7mXcM8rkEEHBAPqf4i+Ibjwr8P9Y1izXN1DEEhOR8juyxq/IIO0sGwRzjHevnT4FaBY698Rka/WRhp1ub6FVbaDKkiBS3cgFs445AzkZB9X/aKgmm+HNq8UUjpDqcTysqkhF8uRct6DcyjJ7kDvXnn7OP/ACUPUP8AsFSf+jYqAPp+vlj49eDrfw54th1ay+W31nzZnjLlis4YGQjI4U71I5PJboMCvqevnz9pmeFrjw1brLGZ0S5d4ww3KrGIKSOoBKsAe+0+lAHb/D7SrHxz8FNCs/EltHfwBGRVb5CoildI9pXBBCqFyDkjOc5OfljVtNm0bWb7S7ho2nsriS3kaMkqWRipIyAcZHoK+t/gxC0Hwk0FHMZJSV/kkVxhpnYcqSM4PI6g5BwQRXzZrqW8nxk1OO8s5721bxBKJrW3BMkyG4O5EAIJYjIGCOT1oA+z6+WP2hLCzsfiHbfY7SC38/T1nm8mMJ5kjSy7nbHVjgZJ5NfU9fMH7R3/ACUPT/8AsFR/+jZaAPR/2ebG4tPhrJNPHsju9QlmgO4HegVIyeOnzIw59PTFeMfG3/kr2u/9u/8A6Tx17/8ABL/kkOhf9vH/AKUSUeNPh34T+IM91CZoLbXbTBluLMoZkLJhBOvVlwFIBwcL8pAJoA4/wF8ZPCJ8DWmh+J5Pss1paLZyJJaPLDcRgFAAF35+QLu3AAljgY6Hwn+EF/4Y8UQ+I9SvNKvrUWjGyksbiR/ncABx8qhlKFx1P3gcdx454z+HPiPwK6Nq9tG1pK+yK8t33xO20HGcAqevDAZ2tjIGa7z9nzxjcWHiNvCkvz2Oo+ZNCAgzHOqZJLZHylEIPXkLjGWyAe769448NeGNRs7DWdXgs7q75iRwx4zjcxAIRc/xNgcHng4P+E78H/8AQ16H/wCDGH/4qvnzxl8HPHF9411u9sNJjurS6vZbiKZLqJQVdi4GHYEEbsHjqDjIwTydr8K/HN5qM9jF4Zvlmh3bmmURRnBwdsjkI3J42k5HIyOaAPsPVdKsdc0u40zU7aO5s7hNksT9GH8wQcEEcggEYIr4Y1XSr7Q9UuNM1O2ktry3fZLE/VT/ACIIwQRwQQRkGvsv4d+G7zwj4E03Q7+SCS6tfN3vAxKHdK7jBIB6MO1eMftH+Hfs2uaZ4hhixHeRG2uCkOAJE5VmcdWZWwAecRdwOAD2P4Y6lDqvwy8O3ECyKiWSW5DgA7oh5THgnjchI9sdOleOfGi4t/HHxK0XwvoEUFxqUGbae6jBbDs3MblVJ2xBWYkZ2734BBrm/hf8UZvA2l65ZTNG9vJbvcWUJhLE3nyIoJBHyFeWyeifLgnDdZ+zz4euNS1zVfGF+3n+Xut4pZgJHed8NI+4ncGCkAnHzeaeeCKAPX3Tw18LvBd1cW9n9i0mzzK6QhpHd2IA5JJZiSqgseOOQBx8oXOqap8SfH1mdXvdtxqV3Faq6plLdHcKFRM/dXcTjPPJJySa+l/jb/ySHXf+3f8A9KI6+YPBPhj/AITHxfY6B9s+x/avM/f+V5m3bGz/AHcjOduOvegD6jg+DXgOHRotMfQo50RxI08krid3C7cmRSDg9dowueQor5s8f+F/+Fe+OX0/TtV8/wAnZc280UmJoMnKq+MbZBgHI6gq3GcD0/8A4Zl/6m7/AMpv/wBtrQ0z9mvR4vN/tbxBfXWceX9khS329c53b89umMYPXPABc+LOpavrPwF07VIljUXqWdxqSxgBRG6hsDcSceaYuhJ/DNeUfCPx3ovgLWdQvtW067uHuLcQwzWpBaMbgWUozKCGwpznI2AAfMa+p5vDmkXPhoeHZ7GOXSRbrbC2clgI1ACjJOcjAw2cggHOea+ePHPwC1fR3kvPC5k1TT1QMbd2H2pMKS3AADjjjb8xLAbTjJAPR/iL4KtPiz4a0rVvDM+mvd7wY7+Z3UPb4bch2qTkPjhhlSGHBJB6DVfhnoviDwVpPhzWDJKdMt4oYb2ACOVSiqpK53ABgvKnI6dwCPlzwp4x8QfDrXLiWw/dTcw3Vldo2xyuRh0yCGU5xyCOR0JB+z7C+t9T062v7OTzLW6iSaF9pG5GAKnB5GQR1oA+GNC0z+2/EOmaT53k/bruK283bu2b3C7sZGcZzjIr6zg+DHw+triKdPDsZeNw6iS5mdSQc8qzkMPYgg96+XPAn/JQ/DX/AGFbX/0atfb9AHzp+0L42mm1SLwhY3EiW9uiy36qSBJI2GRG45CrhuCQS4yMpWv8E/h54U1Twdba/qNnaapqD3EoKy5dbcD5BG8e4oxx84LLn5x6A15J8VNT/tf4oeIbnyfK2XZttu7dnyQIt2cDrszjtnHPWu48K/s/XHiHwvp2sXWv/YJL2ITC3+xiXajHKHcJBnK7W6DGcHkUAb/xt+G/hbTPC8viPTYYNKvopY08iHCR3W4hdqpkBWABf5RyA+Qc5Ef7OXiuaZNQ8KXBkdIUN7asckIu4LInJ4G5lYADqXJPIqSx/Zos47yNr/xPPPajO+OCzETtwcYYuwHOP4T6cda7zwD8KdF8AXEl9aXF3d6hNbrBLNMwCgZBbYgHAZgDyWIwBnrkA+dPi34nTxV8RdQurafzrG2xaWrDbgonUqVzuUuXYHJyGHToPoP4dfC3RfC/h/T5tQ0i0n14ok1xcTRiRopQSwCZLBSmdu5MbtoPpj5Y1aG50vxVfQaoY7+7tb2RLoySOy3Dq5D5bKuQxB5yDz2NfddAHk/xp+Hul6x4X1TxJa2GNdtIllM0UmzzIkPz7wThsJuOfvfIoBIG0+MfBL/kr2hf9vH/AKTyV9J/E7UodK+GXiK4nWRkeye3AQAndKPKU8kcbnBPtnr0r5g+Es9pbfFTw+97LJFEbgorIzqTIyMsY+XnBcqCOhBIbjNAH094E/5mX/sPXX/stdbXJeBP+Zl/7D11/wCy11tZUfgR3Zl/vUvl+SCuS+Jn/JPdU/7Zf+jUrra5L4mf8k91T/tl/wCjUor/AMKXowyv/fqP+OP5ozPjb/ySHXf+3f8A9KI65/8AZx/5J5qH/YVk/wDRUVdB8bf+SQ67/wBu/wD6UR1x/wCzXqfm+Htc0nycfZrtLnzd33vNTbtxjjHk5znnd2xzqcJmfHf4bQ2yXHjTS1jiQvGt/bIgUFmYgz5LdSxjUqF5JLHua83+GHjabwT4xtbp7iRNLuHEV/GCdrRnIDkAEkoTuGBk4IBG419l18efFX4fzeBPEpEXlnSb95JbEqxyigjMbAknK7lGTncCDnOQAD1/9oeeG6+Gml3FvLHNBLqcTxyRsGV1MMpBBHBBHOa6j4MTzXPwk0F55ZJXCSoGdixCrM6qOewUAAdgAK+UBr98PCr+HC0Z09r1b4KV+ZZQhQkH0KkZBz90Yxzn6r+CX/JIdC/7eP8A0okoA4v46/DnSI/D914u0y2jtb+O4V77a5Czq5CZ24I37ipyNucuTk4rzT4Jf8le0L/t4/8ASeSvf/jb/wAkh13/ALd//SiOvAPgl/yV7Qv+3j/0nkoA93+Oes/2R8L72JXnjm1CWOzjeE4xk72DHI+Uojqeud2MYJrxT4K+BLHxr4lvG1i1kuNLsbfc6rNsDSucIrYIbGBIeMcqMnnB+h/iT4Xbxf4D1LSoI42vNgmtNyKT5qHcApJAUsAU3ZGA57ZFfNnwq8V2/wAPvH0k2t208UMkT2NzlSHtiXUlmTGTgpgjqMnqRggH0f4/8L6FqvgbV0vbGxj+z2lzcQ3EkRH2aQgyNICgLDLgM20EtzkNnB+SPCdjb6n4y0OwvI/MtbrULeGZNxG5GkUMMjkZBPSvpP4j/Fvwppvhq+06zurTW7y9t2hFtbyl4tjhlJeRDgADPyhg3I6A7h5R8DPBM3iHxjDrVzbyf2XpT+b5hBCyXAwY0BBByCQ5xkfKAR8woA+m9G0PS/D2nJYaRYQWVquDshTG4gAbmPVmwBljknHJr5Q+Nv8AyV7Xf+3f/wBJ46+v6+QPjb/yV7Xf+3f/ANJ46AO4+D/wf0bX/Di+I/Ea/bYbzelraJI8YjCuVLsVIJYlSAAcAcnJPy1/jd8NfDvhPw9puraBafYs3ZtpovMkk83chZWy7HG3y2GAOd3tXp/wS/5JDoX/AG8f+lElcf8AtKf2j/wj2h+V/wAgz7W/2j7v+u2fuv8Aa+753Tj17UAZ/wCzXrf/ACHNAluP7l7bwbP+AStux/1xGCfoOtZnxx+G2m+HLc+KrK8u3n1HU3FzBNtZd0geTKEAEAFSMHOcjkY50/2Zf+Zp/wC3T/2tVz9pbUpotG0DS1WPyLi4luHYg7g0aqqgc4xiVs8dh07gHlnwq8EWfj3xbJpl/dT29rDaPcuYAN74ZVCgkEDlwc4PTHfI+i/BXwm0HwHrM2qaXd6lNPLbtbst1IjKFLK2RtRTnKDv6153+zL/AMzT/wBun/tavoCgD5//AGmv+ZW/7e//AGjVP9nH/kN6h/17Sf8AoUVdZ+0d/wAk80//ALCsf/oqWuT/AGcf+Q3qH/XtJ/6FFWVXp6ndgf8Al5/gf6H0ZRRRWpwhRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQByPxF8KXfjHw0ml2csMT+eHZpmKjbsdTjAPPzDtXkP8Awzjqf/QStP8Av+3/AMar6MorN003e7OynjZQpqm4xaW11fc+f9J+AWs6NrNjqlvqFg09lcR3EayTOVLIwYA4jBxkeoru/FPhvxv4u8OXeh38nh6O1utm94GmDja6uMEgjqo7V6LRS9l5sr69/wBO4fceXeCvBXjHwHo02l6XPoU0Etw1wzXTzMwYqq4G1VGMIO3rXOeKfgpr/i7xHd65f3umR3V1s3pBLIEG1FQYBjJ6KO9e6UUey82H17/p3D7j5z/4Zx1P/oJWn/f9v/jVdH4K+E3iXwHrM2qaXd6TNPLbtbst1JIyhSytkbUU5yg7+te00Uey82H17/p3D7j5z/4Zx1P/AKCVp/3/AG/+NUf8M46n/wBBK0/7/t/8ar6Moo9l5sPr3/TuH3HjXiv4VeI/F2naNYXUui2lrpETQ2qWkswwhCDBLqxOBGv65zXMf8M46n/0ErT/AL/t/wDGq+jKKPZebD69/wBO4fceXeCvBXjHwHo02l6XPoU0Etw1wzXTzMwYqq4G1VGMIO3rXOeJ/gjq/ijXJtXmbRbK4n5mWyklRJH7uQyN8x74xnrjJJPulFHsvNh9e/6dw+4+c/8AhnHU/wDoJWn/AH/b/wCNVJJ+zrqLpCq3dhGUTazLcSZkO4nc2YyM4IHGBhRxnJP0RRR7LzYfXv8Ap3D7jwvwr8DdU8OeJbDVPt1m6QTxu6+cxJVXVjj92OePWvdKKKqEFG5liMVKsoppJLaytuFFFFWcxw3xR02HWdG0jS7hpFgvdWht5GjIDBXV1JGQRnB9DXmN9+zReR2cjWHieCe6GNkc9mYkbkZywdiOM/wn04616147/wCZa/7D1r/7NXW1lD45fI7q/wDutH/t78z5kh+GXxglc6G+pXcOlhGtvMfVybUxBSANisW2EDAGzoRkDnHV+Ef2eLPTrxL3xFqP2ySGWGWK3tlHkttAZ1kDqd6lsrgYyoyeWwvuFFanCFfOn7S2pQy6zoGlqsnn29vLcOxA2lZGVVA5znMTZ47jr2+i6+dP2ltShl1nQNLVZPPt7eW4diBtKyMqqBznOYmzx3HXsAU/Bfwo8Ur4QtfGPhzX57DWZojLDYGExeaokyqly2GVwqsAy7Tlc8c1oX3hH42eNrOTSPEF5BbWBxK3nywIkjKRhT5Clj1zgjb8ueoFej/BL/kkOhf9vH/pRJXoFAHF+AfhnovgC3kazMl1qE6Ks97MBuIAGVQD7iFhnHJ6ZJwMHxJ0HxV4j0a0svC2px6ZOtx5s1ybyWBtoUgIPLUkglsnJGNo4OeO0ooA+YP+GcfGH/QS0P8A7/zf/Gq9/wDB1v4ptND8jxde2N7qSyttnswQHjOCNw2qNwO4cADAXvk10FFAHzp4z+A2r6Xqiap4GlknQ3G+O184RS2mMFSkjMNwDA4OQw+X73LDHg8WfHC2t4oEtvEBSNAimTRQ7EAY5ZoiWPuSSe9fUdFAHi/wj+Feo6NrFx4p8WpOutiVxbxvOsn31+eZmVjuZtzLg9OSckgi58XNL+JOtXp03wqkj6Dc2SpdIksEZaTe24bnIfBXYCAcEZHc165RQB8eSfDT4keF3h1ODRdSgnV9kcunTCWVCVOf9SxYDGQT05x3r3vU/wDhZ9x8L9NksfItfF0co+2RjyT5sYLrxu3R7iPLc8gcNjH3a9IooA+QP+FQfEfU/wDT5dCnkkuv3zPcXcQkYtyS4d9wY553c56819B/CPS/FGi+ChpvipJEuLa4ZLVHlSQrb7V2jchPAbeACcgYHQCu8ooA8v8Ail8Ibfxxt1PSngstdXarySZEdyg4xJgEhgOjAHgbTxgr4p4af4r+ELee30LSvEFrBO4eSM6U0i7gMZAeMgHGASMZwM9BX13RQB8wP8FviL4is7rUtdvN1/bxFLaC9vfPmmwQQofJVVO58Zb7w5AB3DL8H/Db4m2WqWGt6PpUmnTh2WO5uzEhhDZRmaJ8tjBb+AnHIB4r6zooA5/xp4Ts/Gvhe60W8fyvNw8M4QO0MinKsAfxBxglSwyM5r5gtbrxr8FPFE4Nv5Xm7omWZGe0vVUZDKRjdjcCCCGXODjLKfr+igD5Y1DxL8Rfi/pxsbaxna0MqobewtfLtJGQM7GWd34YZTCE7TweGA3ex/Cz4Zr4At9Sa7NpdahPcMsV7EG3G2AXapB+4SwJIGR05OBj0SigDD8YeGofGHhO/wBBnuJLdLtFAmQAlGVg6nB6jcoyOMjPI618uW+neNfhB45s5V0/dfTb4bZF3Sw3yk7NoCEFuSrBDhgdhIHFfX9FAHzZ/wAJp8Svim9z4PgsrTTAztFfzxW88QiAVyYpnJcoGKlcYBJG3oSDzd98CvH1peSQQ6XBexrjE8F3GEfIB4DsrcdOQOnpzX1vRQB5PLrHxJ8JfCrSpZdN/tfxH9r8iaFrd7h0gw+0uYnO9vlT58j7wBBbLHwC/wDCXjjU9Rub+88Ma5JdXUrzTP8A2bKNzsSWOAuBkk9K+16KAPI/hHrvxI1bWdQi8YWl2unpbho5buxFsyy7hhVwq7gV3E8HG1eRnnr/AA//AMlC8Y/9uX/oo11tcl4f/wCSheMf+3L/ANFGsqnxR9f0Z3YT+DX/AMC/9LgdbRRRWpwmT4p/5FDWv+vCf/0W1fOHj7SNK1fSLPUdF1v/AE82lnHfaUYpVE0sUZj8wOfk3KpCgcDAYg5OD9S0VnKEubmi/wAP+Cjto4iiqLo1oN630kl0t1jI+a/AfxQ8QeDPDkOhzeGoL+1t932d0uxC43OztuJ3BuW4wFxjvWP411/xP8R9RsoNRuLHTNMG+SOAPIY7dsuB5pVWZ5NoADKCoDDAXLgfVlFK1Xuvu/4I/aYH/n3P/wADX/ys8u0DxV4K8D+Cl03RbmO6uLa3L7Ft5YTe3AXkszKdpdgBkkhRgdAK8M8e6t4g8feIRq15pcFr5cX2eGKGQHEYdmXcS3zN85BIABx0FfYlFFqvdfd/wQ9pgf8An3P/AMDX/wArPkvwR408aeAtOubDTLCxntZ5fOKXYDbHwASCrqeQF65+6MY5z7HqfjXSvFEXhOO3l2agdVs557Tax8k87l3lQGwzYyOvWvUaKUoVJKza+7/gmtHFYSjNVIU5XXeat/6QgooorY8wyfFP/Ioa1/14T/8Aotq4fxf4OuPG3wX0uwseb+3tba6tULhFkdYsFSSO6s2OnzbckDNejahZx6jpt1YzMyxXMLwuUOGAYEHGe/NcvF8P44IUhh8T+JY4o1CoiX+FUDgAALwKykpKfMlfQ9CjKjPDOlUnyvmvtfpY+UNJ1rxL8P8AXJ5LJp9K1PyvJmSe2XeEba+CkinGcKen869Ug+KXxeuriLUrfwpJNZy248uGPSJ2gfJ3CUMDuJI4+9tx2zzXr3/CCf8AU1+KP/Bj/wDY0f8ACCf9TX4o/wDBj/8AY0c8/wCX8SfYYX/n9/5Kz5h8R6H4+8U+ILzWtS8Laybu6cM/l6ZKqqAAqqBt6BQBzk8cknmvc/gzrPjWezOh+I/D89pYadaJHbXk9u1u5wcLGVbG/wCX+JQMbPmyWBrq/wDhBP8Aqa/FH/gx/wDsaP8AhBP+pr8Uf+DH/wCxo55/y/iHsML/AM/v/JWeI/GPUfGXijVLq1m8J3cGi6RcTGG7XT5GLovBkaYrgIQpbC4GCM7toNaHwD1OHQvEF3pslnqRg8QPjTLyS2Eccq24lLkncRnBAwpbB4J7169/wgn/AFNfij/wY/8A2NH/AAgn/U1+KP8AwY//AGNHPP8Al/EPYYX/AJ/f+Ss4j43+A9X1240rxF4f06O8vLBHW5jOHZ41O9MRPlXAPmZXBLbwMMOnmngz44+IPCtnHp91bQarYrLLKxmZlnJclj+85z85LEsrE5Iz0x9Bf8IJ/wBTX4o/8GP/ANjWHP8ABHwrdXEtxcS3808rl5JJGiZnYnJJJjySTzmjnn/L+Iewwv8Az+/8lZ5z4x/aEuNV0P7B4csJ9NuLiJRcXckoLxE53pFj8MSHB64VTgh/gHwO+g6HoWs6tpE9lrM/iOGGNpyyP9m4ODGT8vzox5AP4EV6Xpvwg0HRrhrjS9Q1axnZCjSWsyRMVyDglUBxkA49hWongG3+2Wlxca9r139lnS4jjurwSJvU5BIK/h+JqZ88o25fxN8MsLQqqp7W9r/ZfY62iiitzygrkvhn/wAk90v/ALa/+jXrra4HQtG8deH9Gt9LtH8OPBBu2tKZyx3MWOcADqT2rGbampW6P9D0cNGNTC1KTkk3KL1dtEpp/mje8aeE7Pxr4XutFvH8rzcPDOEDtDIpyrAH8QcYJUsMjOa+TPB/hKbVfibYeGNQhjV0vWjvIXkIG2IlpU3JnnajAEHrjkda+pv+Lh/9Sv8A+TFU7PSfGOn3F1cWWn+Dbae7ffcyQwzI0zZJy5AyxyxOT6n1p+18mR9R/wCnkPvO5rg/i/4a1LxR8Pru00u4kSe3cXTW6Bj9qVAx8rC8kk4YDByyqOOov/8AFw/+pX/8mKP+Lh/9Sv8A+TFHtfJh9R/6eQ+8+UfA3jO+8C+JY9YsY45gUMNxA/AmiJBK5xlTlQQR0IGQRkHtPG/xB8QfFbQ7iCw8NeRpOk7L66eNmmeMjemS+FG3Dk7duflZs4Bx6te/D/VNR1SPUrvw54Glu0eRy5gmAkZ/vGRRxIe+XBweRg10EEHj21t4re3i8KQwRIEjjjWdVRQMAADgADjFHtfJh9R/6eQ+8+UfCnjTXfBV5cXOiXfktcRGKVHUOjcHaxU8blJyD+HIJB+k/iN8UR4M8NW0MtpJB4k1GyLxW0cscgspCAMue4Vi20hSHMZHA5qex8J6/pl5HeWHh/wFaXUedk0FlJG65BBwwGRkEj8aPEnhbxV4u06Ow1yz8L3drHKJlTzbuPDgEA5Qg9GP50e18mH1H/p5D7z5V0LU/wCxPEOmat5PnfYbuK58rdt37HDbc4OM4xnBr6/+H/xE0v4g6dPNYwz211abBdW8wzsLAkFWHDLkMAeD8vIGRWF/wgmp/wDQqfDv/wAFzf8AxNamm6T4x0a3a30vT/BtjAzl2jtYZolLYAyQoAzgAZ9hR7XyYfUf+nkPvMz4n/FnTfBlvdaRZvJP4gktz5QiClbVmA2tITkZwdwXBzgZwGBr5o8H+JZvB/iyw16C3juHtHYmFyQHVlKMMjodrHB5wccHpX1NqeheK9b8r+1tK8E3/k58v7Xbyy7M4zjcDjOB09BWf/wgmp/9Cp8O/wDwXN/8TR7XyYfUf+nkPvOh0TX9F+KPgq+axa7js7tJrG4V1CSxErgj+Jc7XBBGRyM85FfOGgXmr/Bb4mr/AGxaSFFQxXKQ4IuLdjw8bMORuVWHQ5QqSvzY+hrGw8b6ZZx2dhaeELS1jzshgjmjRckk4UcDJJP41l+JfB/iTxhbwQa9pvhS8SBy8RL3SMhIwcMpBweMjODgego9r5MPqP8A08h95Lf/ABv8B2el/bodVkvSXKLb29u4lYjbnhwoAwwOWIBwwGSCK8QEHiP47/EF7hYo7W2iRUeQLujsrcElQTwXcksQOCxJ+6o+Xq4P2ddRhuIpXu7CdEcM0UlxIFcA/dO2MHB6cEH0Irv/AA14P8SeD7eeDQdN8KWaTuHlIe6dnIGBlmJOBzgZwMn1NHtfJh9R/wCnkPvO90rSrHQ9Lt9M0y2jtrO3TZFEnRR/MknJJPJJJOSa+QPGMl94W+Meq3zQxi7tdYa+iSQ7lYGTzYydp6FSpxkHnHBr6e/4uH/1K/8A5MVy/iz4e69412yazYeF5LqOJoorqKS5jkjB9wcNg8gMCASeOTk9r5MPqP8A08h95Qg/aQ8KtbxNcaVrMc5QGRI44nVWxyAxcEjPfAz6CvFPiV43/wCE98Wtqsdr9mtYohbWyMcuY1ZmDPzjcSxOBwOBzjJ9T0b4B32kail5Kui6lswVhvpZmjDAgglUVd3TGGypBOQa6fVvhzqWt2cFre+HfBPk2/EIgjuICgyx2ho9p25djtzjJzjNHtfJh9R/6eQ+8wPgd8SdNOl6R4Gls7tdQDziKZdrRMvzzZY5BU/eGAD0BzzxwFp4zbwL8c9e1do5JbRtTvILuKPbueJpWzjI6hgrdRnbjIBNezaJ4G1rw55DaToPgm3mg3eXceVO8y7s5/etlzwSOT046Vj658Ir3xBeXt7d6P4XjvrzcZLiCe7jIcjG8KDs3Z55U5PJBycntfJh9R/6eQ+8ofEH42eFNV8FTaZpttJqc+p27RyRTxmMWZK/KzblIZ1YqQFyMqTuGBngPgRpV9e/E+wv7e2kktLBJXuph92IPE6Lk+pY8DrwT0BI6uD9nXUYbiKV7uwnRHDNFJcSBXAP3TtjBwenBB9CK9H0Dw54s8LaWum6LY+FLS0Dl9im5Ysx6lmYksegySeAB0Ao9r5MPqP/AE8h956DRXJf8XD/AOpX/wDJij/i4f8A1K//AJMUe18mH1H/AKeQ+862sPxjoC+KfB2q6Kyxl7q3ZYvMZlVZR80bErzgOFPfp0PSs7/i4f8A1K//AJMUf8XD/wCpX/8AJij2vkw+o/8ATyH3nx7YaNeX3iG20PZ9nvp7tLPZcAp5cjOEw4xkYJ54yPSvtvw5oFj4W8P2ei6asgtLVCqeY25mJJZmJ9SxJ4wOeABxXlWm/CDVNK8Yt4ktrPw2Jw5khtGaY28EnHzomMgg5IBJCk8AYXHef8XD/wCpX/8AJij2vkw+o/8ATyH3kvxF8PXHir4f6xo9m2LqaIPCMD53RlkVOSANxULknjOe1fInhzWr7wV4xs9UWCRLvTrgiW3kGxiOVkjO4HaSpZc4yM+or62/4uH/ANSv/wCTFcX4t+FeseM7iW81Gy8NxahIiob21kuUkwp4yOUY4+XLKTjA7DB7XyYfUf8Ap5D7zpbL4y+A7zS5L867HbiJI2lgnidZVL8bQmCXIPXZuA65xzXlnxB+Ij/FLUdP8FeEoZxa3N2BJcShl88gkA7VyRCB+8JYZ4Bwu3l//DOOp/8AQStP+/7f/Gq6vwl8K9Y8GXEV5p1l4bl1CNGQXt1JcvJhjzgcIpx8uVUHGR3OT2vkw+o/9PIfeZnxysG0P4R+GtFAkuEs7iCA3QVVUmOB0GV3ZBbkjGQNpyRxk+GHxp0Cy8HWukeJ7ySzu9PQQRSmBpFmiGdmPLU7SqgKcjnAOSScdf4k8LeKvF2nR2GuWfhe7tY5RMqebdx4cAgHKEHox/OuE1L9n28vrhZbcaTp6BApitbmdlJyfmPmK5zzjrjgcdcntfJh9R/6eQ+884+JPiiz+IPjlL7Q9Kni82KK2VDGDNcyZOGKpnLchAMsSFX6D630LTP7E8PaZpPned9htIrbzdu3fsQLuxk4zjOMmvLvBXwu8QeA7ia70tdCmvJUaNrm6muGYRkqdgChVxlAc4z15xxXaf8AFw/+pX/8mKPa+TD6j/08h958g6TeTeGvFVje3FpJ5+mXscsltJmNt0bglDkZU5XHTj0r6Pg/aK8GzXEUT2mswI7hWlkt4yqAn7x2yE4HXgE+gNReM/hbrvjh0n1C38N214r7mvLLzY5ZBtC7XJUhhgLjIyMcEAkHM8PfA6+0C8N09r4e1ST+Aak00qJwQfkCqrZz/EDjAIwaPa+TD6j/ANPIfecZ+0D4euNM+IH9sM2+11aJXjOANrxqsbJ1ycAI2cD7+OcGu7+E3xjsbzRk0fxbrEcOpwvsgurr5VniC5G+QnG8YIJbbnK/eYmun8U+G/G/i7w5d6HfyeHo7W62b3gaYONrq4wSCOqjtXnH/DOOp/8AQStP+/7f/GqPa+TD6j/08h956Z4n+Mng7w7ZzGHVINUvhF5kNtZP5gkJOADIoKLyMnJyBzg5APL/ALPlnqlxZ+I/E+pHd/bF2pVym0yuhcyOBgLtLSYG3urDAxXN/wDDOOp/9BK0/wC/7f8AxqvW4IPHtrbxW9vF4UhgiQJHHGs6qigYAAHAAHGKPa+TD6j/ANPIfefMvxW0C+8P/EbV1vljxfXEl9bsjbg0UkjEH1BByCD3BxkYJ9b+G/xx0ZPC8Vj4v1KeHUrT92LqWJ5vtKZO05RSQwGFO7rgHJJOOv1/w54s8U6W2m61Y+FLu0Lh9jG5Uqw6FWUgqeoyCOCR0JrzT/hnHU/+glaf9/2/+NUe18mH1H/p5D7y58Y/i3oWq6DeeFdF/wCJh9o2ie9RyscTJKrALkfvM7DyCFwQQW5x454O1ex0DxjpWr6lZyXdpZ3CzPFG+1sj7rD1KthsEgHbgkA17Hpv7Pt5Y3DS3A0nUEKFRFdXM6qDkfMPLVDnjHXHJ46Yrwfs66jDcRSvd2E6I4ZopLiQK4B+6dsYOD04IPoRR7XyYfUf+nkPvPT/AIXalDrOjavqlusiwXurTXEayABgrqjAHBIzg+prua5XwD4YuPCmhz2FwLVd900saWruyIhVQFBf5uNvfPbk11VOkmoK5OYTjPEycXdafkFcl8TP+Se6p/2y/wDRqV1tcl8TP+Se6p/2y/8ARqUq/wDCl6MrK/8AfqP+OP5o80+MfxV8Nar4QvPDejXX9o3V1KscssQYRwiORWzuIw+SuBtyMZOegbA+B3xE0bwjp2sWHiHVfslrJLHNap9neTLkMJDlFJ6LH19OO9bcH7M0K3ETXHiuSSAODIkdgEZlzyAxkIBx3wcehrc/4Zx8H/8AQS1z/v8Aw/8AxqtThPYKx/E/hjS/F2hzaRq8Hm28nKsvDxOOjoezDJ/MgggkHUgghtbeK3t4o4YIkCRxxqFVFAwAAOAAOMVJQB8MeJ/DGqeEdcm0jV4PKuI+VZeUlQ9HQ91OD+RBAIIH1P8ABL/kkOhf9vH/AKUSVoeN/hr4f8e/ZpNVSeG6t+EurRlSQpz8hJUgrk55HBzgjJzueHNAsfC3h+z0XTVkFpaoVTzG3MxJLMxPqWJPGBzwAOKAOD+L3xC0bRfC+teH4r+B9duLQRC0MbvhJSFbJUYVthZgCR/CcEEZ8E+FfiXSPCfjyz1XWbeSS3VHiWZCc2zONvmbR98BSwI9GJAJAB+h/FHwV8IeKdUfUporuwu5XZ53sZVQTMccsrKwB4JyoGSxJyTUdj8CvANpZxwTaXPeyLnM893IHfJJ5CMq8dOAOnrzQB2B8T6W/hKbxPaz/bNMjtHuw8HJdEUlgAcYbgjBxgjBxXy54Vh8OfEHxjqt/wCO/EUmmT3b+ZCqPsV2OSR5sgZURFUKFbrlQDxg/QfizXvC3wx8DLZy2UBtDE1vaaUoB+05HzKc5yvOXZs/e5yzAH5k0D4aeMPFGlrqej6LJPZs5RZWmjiDkddu9gSM8ZHGQR1BoA9v0rwX8E9MS336ro19PA+/zrvWEYud2RuRXCEdsbcEDnPOfVNM13R9b83+ydVsb/yceZ9kuEl2ZzjO0nGcHr6GvkTUvhL480q3We48NXbozhALVkuGzgnlY2YgcdcY6eorn9X8N614fS0fWNLu7EXaM8P2iIoWCttPB5BB7HnBU9GBIB9v6rqtjoel3Gp6ncx21nbpvllfoo/mSTgADkkgDJNfHHxL1+x8UfEPV9Y0xpGs53jWJ3XaXCRqm7HUAlSRnBwRkA8V6n8Ml/4W34G1Twz4sub66/s27iuYL77RmZd4fC5YHONr8tu4kwMbRR/wzL/1N3/lN/8AttAHYfArXNLu/hxpukQX8D6laeeZ7XfiRAZmYNtPJXDr8w4ycZzkVxH7QnjTTtSisvDOm3cFzJaXby3wRWJhkVdqKG+6fvyZAyQVwcEEHq/AnwNh8GeLLbXpdfkvntkcRQraiEbmUpljvbI2s3AxzjnjBw779mizkvJGsPE88FqcbI57MSuvAzlg6g85/hHpz1oAwP2ePE+l6Prmp6Rfz+Rcar5AtWfhHdN/yE9mO8Y9cYzkgHX/AGmv+ZW/7e//AGjUkH7M0K3ETXHiuSSAODIkdgEZlzyAxkIBx3wcehr2DxX4S0jxnox0vWYZJIA/mRtHIUaOTayhxjgkBjwQR6g0AfMHwY8W6R4O8azXutTSQWk9lJb+csZcIxZHBYLk4+QjgHkjtkj6H8NfFTwp4s8QT6NpV7I1wiB4WljMa3IxlvL3ckr3BAPBIyASPGJ/2b/FS3Eq2+q6NJAHIjeSSVGZc8EqEIBx2ycepr0f4Z/BqHwNqh1m/wBQjv8AUDbiNESAKluzffKsclj/AAhsLwWyPm4AOE/aD8b/AG7U18Gw2u2PT5Y7m4nc8vIY8qqgH7oWTknkk9gMtW/Z91KGx8Si3lWQvfiW3iKgYDBFly3PTbGw4zyR9R2HxV+D+seNvFser6NNpVtGbRIp/PZ0eSRWb5jtQ5+UoMk5+XHQCsv4O+CdR8OeMjbeI9P+z30FtJe2y+cr7d22LdlGIPBkGD9cdDWVXp6ndgf+Xn+B/oe+UUUVqcIUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRXJf8LM8If9Bf/wAlpf8A4iplOMfidjejha9e/sYOVt7Jv8jraK5L/hZnhD/oL/8AktL/APEUf8LM8If9Bf8A8lpf/iKj29L+Zfeb/wBl47/nzP8A8Bf+R1tFcl/wszwh/wBBf/yWl/8AiKP+FmeEP+gv/wCS0v8A8RR7el/MvvD+y8d/z5n/AOAv/I62iuS/4WZ4Q/6C/wD5LS//ABFH/CzPCH/QX/8AJaX/AOIo9vS/mX3h/ZeO/wCfM/8AwF/5HW0VyX/CzPCH/QX/APJaX/4ij/hZnhD/AKC//ktL/wDEUe3pfzL7w/svHf8APmf/AIC/8jraK5L/AIWZ4Q/6C/8A5LS//EUf8LM8If8AQX/8lpf/AIij29L+ZfeH9l47/nzP/wABf+RF8RbqGys9Au7h9kEGtW8kjYJ2qock4HPQVL/wszwh/wBBf/yWl/8AiKP+FmeEP+gv/wCS0v8A8RR/wszwh/0F/wDyWl/+IrJ1I8zcZrX+u53xwlZ0Y062GqPlvqrrf1gw/wCFmeEP+gv/AOS0v/xFH/CzPCH/AEF//JaX/wCIo/4WZ4Q/6C//AJLS/wDxFH/CzPCH/QX/APJaX/4ij2v/AE8j/XzF/Z//AFC1fv8A/uZl+I/ito9n4fvJ9Bnj1DVFQC2tpI5I1ZiQMklQMAEtjIzjGRnNfNniS48W+LtRjv8AXH+13UcQhV8RR4QEkDCYHVj+dfU//CzPCH/QX/8AJaX/AOIo/wCFmeEP+gv/AOS0v/xFHtf+nkf6+Yf2f/1C1fv/APuZ86eEvGfj/wAGW8Vnpzxy6fG7OLK6CPHlhzg5DqM/NhWAzk9zn6DsPil4YuNOtpry9+yXUkSPNb+VLJ5TkAsm4JhsHIyOuKsf8LM8If8AQX/8lpf/AIij/hZnhD/oL/8AktL/APEUe1/6eR/r5h/Z/wD1C1fv/wDuYf8ACzPCH/QX/wDJaX/4ij/hZnhD/oL/APktL/8AEUf8LM8If9Bf/wAlpf8A4ij/AIWZ4Q/6C/8A5LS//EUe1/6eR/r5h/Z//ULV+/8A+5h/wszwh/0F/wDyWl/+Io/4WZ4Q/wCgv/5LS/8AxFH/AAszwh/0F/8AyWl/+Io/4WZ4Q/6C/wD5LS//ABFHtf8Ap5H+vmH9n/8AULV+/wD+5h/wszwh/wBBf/yWl/8AiKP+FmeEP+gv/wCS0v8A8RR/wszwh/0F/wDyWl/+Io/4WZ4Q/wCgv/5LS/8AxFHtf+nkf6+Yf2f/ANQtX7//ALmH/CzPCH/QX/8AJaX/AOIo/wCFmeEP+gv/AOS0v/xFdbRWtqvdfd/wTh9pgf8An3P/AMDX/wArOS/4WZ4Q/wCgv/5LS/8AxFH/AAszwh/0F/8AyWl/+Irray5fEmhQTPDNrWnRyxsVdHukDKRwQQTwaT9ot5L7v+CXD6pU0hRm/Sa/+VmN/wALM8If9Bf/AMlpf/iKP+FmeEP+gv8A+S0v/wARWv8A8JT4e/6D2l/+Bkf+NH/CU+Hv+g9pf/gZH/jS5pfzL7v+Caexof8APip/4Ev/AJWZH/CzPCH/AEF//JaX/wCIo/4WZ4Q/6C//AJLS/wDxFa//AAlPh7/oPaX/AOBkf+NH/CU+Hv8AoPaX/wCBkf8AjRzS/mX3f8EPY0P+fFT/AMCX/wArMj/hZnhD/oL/APktL/8AEUf8LM8If9Bf/wAlpf8A4itf/hKfD3/Qe0v/AMDI/wDGj/hKfD3/AEHtL/8AAyP/ABo5pfzL7v8Agh7Gh/z4qf8AgS/+VmR/wszwh/0F/wDyWl/+Io/4WZ4Q/wCgv/5LS/8AxFdVFLHPCk0MiyRSKGR0OVYHkEEdRT6q1Xuvu/4Ji54JaOnP/wADX/ys5L/hZnhD/oL/APktL/8AEUf8LM8If9Bf/wAlpf8A4iutrOutf0ayuHt7vV7CCdMbo5blEZcjIyCc9CDSftFvJfd/wSofVJu0KU36TX/ysw/+FmeEP+gv/wCS0v8A8RR/wszwh/0F/wDyWl/+IrX/AOEp8Pf9B7S//AyP/Gj/AISnw9/0HtL/APAyP/GlzS/mX3f8E09jQ/58VP8AwJf/ACsyP+FmeEP+gv8A+S0v/wARR/wszwh/0F//ACWl/wDiK1/+Ep8Pf9B7S/8AwMj/AMaP+Ep8Pf8AQe0v/wADI/8AGjml/Mvu/wCCHsaH/Pip/wCBL/5WZH/CzPCH/QX/APJaX/4ij/hZnhD/AKC//ktL/wDEVr/8JT4e/wCg9pf/AIGR/wCNH/CU+Hv+g9pf/gZH/jRzS/mX3f8ABD2ND/nxU/8AAl/8rMj/AIWZ4Q/6C/8A5LS//EVU8GanZ6z4y8WX9hN51rL9j2PtK5xGwPBAPUGui/4Snw9/0HtL/wDAyP8Axo/4Snw9/wBB7S//AAMj/wAaWrknKS0/y9SrRhSnClQmnJWu3fqnsoLt3Naisn/hKfD3/Qe0v/wMj/xo/wCEp8Pf9B7S/wDwMj/xrbnj3PP+q1/5H9zNaisn/hKfD3/Qe0v/AMDI/wDGj/hKfD3/AEHtL/8AAyP/ABo549w+q1/5H9zNaisn/hKfD3/Qe0v/AMDI/wDGj/hKfD3/AEHtL/8AAyP/ABo549w+q1/5H9zNaisn/hKfD3/Qe0v/AMDI/wDGj/hKfD3/AEHtL/8AAyP/ABo549w+q1/5H9zNaisn/hKfD3/Qe0v/AMDI/wDGj/hKfD3/AEHtL/8AAyP/ABo549w+q1/5H9zNaisn/hKfD3/Qe0v/AMDI/wDGj/hKfD3/AEHtL/8AAyP/ABo549w+q1/5H9zNaisn/hKfD3/Qe0v/AMDI/wDGj/hKfD3/AEHtL/8AAyP/ABo549w+q1/5H9zNaisn/hKfD3/Qe0v/AMDI/wDGj/hKfD3/AEHtL/8AAyP/ABo549w+q1/5H9zNaisn/hKfD3/Qe0v/AMDI/wDGj/hKfD3/AEHtL/8AAyP/ABo549w+q1/5H9zNaisn/hKfD3/Qe0v/AMDI/wDGj/hKfD3/AEHtL/8AAyP/ABo549w+q1/5H9zNaisn/hKfD3/Qe0v/AMDI/wDGj/hKfD3/AEHtL/8AAyP/ABo549w+q1/5H9zNaisn/hKfD3/Qe0v/AMDI/wDGj/hKfD3/AEHtL/8AAyP/ABo549w+q1/5H9zNaisn/hKfD3/Qe0v/AMDI/wDGj/hKfD3/AEHtL/8AAyP/ABo549w+q1/5H9zNaisn/hKfD3/Qe0v/AMDI/wDGj/hKfD3/AEHtL/8AAyP/ABo549w+q1/5H9zNaisn/hKfD3/Qe0v/AMDI/wDGj/hKfD3/AEHtL/8AAyP/ABo549w+q1/5H9zNaisn/hKfD3/Qe0v/AMDI/wDGj/hKfD3/AEHtL/8AAyP/ABo549w+q1/5H9zNaisn/hKfD3/Qe0v/AMDI/wDGj/hKfD3/AEHtL/8AAyP/ABo549w+q1/5H9zNaisn/hKfD3/Qe0v/AMDI/wDGj/hKfD3/AEHtL/8AAyP/ABo549w+q1/5H9zNaisn/hKfD3/Qe0v/AMDI/wDGj/hKfD3/AEHtL/8AAyP/ABo549w+q1/5H9zNaisn/hKfD3/Qe0v/AMDI/wDGj/hKfD3/AEHtL/8AAyP/ABo549w+q1/5H9zNaisn/hKfD3/Qe0v/AMDI/wDGj/hKfD3/AEHtL/8AAyP/ABo549w+q1/5H9zNaisn/hKfD3/Qe0v/AMDI/wDGtampJ7MidKdP44teqCiiimZhRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAVyXxM/5J7qn/bL/ANGpXW1zHxDtbi98C6lb2lvLPO/lbY4kLs2JUJwBz0BNZ1v4cvRnblrSxtFv+aP5o6eiuS/4Tv8A6lTxR/4Lv/sqP+E7/wCpU8Uf+C7/AOype2h3H/ZuK/l/Ff5nW0VyX/Cd/wDUqeKP/Bd/9lR/wnf/AFKnij/wXf8A2VHtodw/s3Ffy/iv8zraK5L/AITv/qVPFH/gu/8AsqP+E7/6lTxR/wCC7/7Kj20O4f2biv5fxX+Z1tFcl/wnf/UqeKP/AAXf/ZUf8J3/ANSp4o/8F3/2VHtodw/s3Ffy/iv8zyH9pbUoZdZ0DS1WTz7e3luHYgbSsjKqgc5zmJs8dx17et/DbxZoXinwlaf2In2ZbGKO3lsHcs9rtXCqSeWXA4bvjsQQOe8ZtovjrS0sdY8H+KwYn3w3EFgFlhPGdpJIwQMEEEHg9QCPGJPht4q0fVIbrw4muh40yLlrCW0lRjkELsL8bT13DOSMep7aHcP7NxX8v4r/ADPrivA/2h/Gdt9nt/B9vHHLcb0u7qU7G8kYbag4JVzncT8p2kDkOa4/+xvi/wD9BLxR/wB/7v8AwqPwv8LLmPVEn8UaPrslnE6t9msbB2M45yrOxUoM7egJIJ5U4NHtodw/s3Ffy/iv8z0j9nDRvsnhLU9XdJ0kv7sRLvGEeOJeGXjn5nkBOSPlxwQa9orirHxZZ6ZZx2dh4K8Q2lrHnZDBpQjRckk4UHAyST+NWP8AhO/+pU8Uf+C7/wCyo9tDuH9m4r+X8V/mdbRXJf8ACd/9Sp4o/wDBd/8AZUf8J3/1Knij/wAF3/2VHtodw/s3Ffy/iv8AM62iuS/4Tv8A6lTxR/4Lv/sqP+E7/wCpU8Uf+C7/AOyo9tDuH9m4r+X8V/mdbRXJf8J3/wBSp4o/8F3/ANlR/wAJ3/1Knij/AMF3/wBlR7aHcP7NxX8v4r/M62uS/wCavf8AcB/9r0f8J3/1Knij/wAF3/2VVNGvLjWfiM+qf2RqljarpJt919bGLL+cGwOo6H17GonUjJpLudOHwlajGpKorLlfVHcUUUV0HkBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFfF/irwjrVraQ+IzZSS6TeJuFzEpZYireWVkOPkJYDGeDuGCTkD7QryKfU/7I/ZsvbnyfN32k9tt3bcedM0W7OD035x3xjjrWT/AIq9H+h3U/8Acan+OH5VD550rxl4m0NLeLTNf1K2gt33xQJct5Sndu/1edpBOSQRg5Oc5r7X0nUodZ0ax1S3WRYL23juI1kADBXUMAcEjOD6mvgyvpv9nXX21DwdfaLK0jPpdwGjyqhVilywUEck71lJz/eHPYanCeyVHPPDa28txcSxwwRIXkkkYKqKBkkk8AAc5qSvO/jbr7aD8Mr5YmkWfUXWxRlVWADglw2egMauMjJyR06gA8Q1X47+OL3VLi4sNRj0+0d8xWqW8UgiXsNzoSx7k+pOABgDpPh58QPin4k8UaZ80+o6Q12IryQ2EawomMvmRVXayqdwG7k7Rg5wfD6+v/gl/wAkh0L/ALeP/SiSgD0CiiigAooooAKKK+SPhz4s8a6n8RdCs4vEGq3ayXa+dDPetIjQjmXKucHCBz68cc4oA+t6+TPiT8IJvh9o1pqi6zHqEE1x9ndTbmFkYqWUj5mBGFbPIxx1zx9Z14/+0d/yTzT/APsKx/8AoqWgDkP2a9M83xDrmredj7NaJbeVt+95r7t2c8Y8nGMc7u2Ofo+vn/8AZl/5mn/t0/8Aa1fQFABRRRQAUUUUAFFFFABXJfEz/knuqf8AbL/0aldbXJfEz/knuqf9sv8A0alZV/4UvRndlf8Av1H/ABx/NHW0UUVqcIVwnhHRdK1GbxJNfaZZ3Uq65dKHngV2A+U4yR05P513dcl4E/5mX/sPXX/stYzSc438z0MLOUMNWcXZ+7+Zr/8ACLeHv+gDpf8A4Bx/4Uf8It4e/wCgDpf/AIBx/wCFa1FackexzfWq/wDO/vZk/wDCLeHv+gDpf/gHH/hR/wAIt4e/6AOl/wDgHH/hWtRRyR7B9ar/AM7+9nyb8WfGmn6l4gfSvDFvp9tpdk+PtdhEqNdSY5O8AHYCSABwcbsnK4oeKfBfifwHrWkwazfJPFevmOS2uXdG2sAykMAcjcvbHzdTzjB8Cf8AJQ/DX/YVtf8A0atfYvjCCG58G6yk8UcqCzlcK6hgGVSynnuGAIPYgGonCPK9DfDYms68E5vddX3JPC3/ACKGi/8AXhB/6LWtasnwt/yKGi/9eEH/AKLWtarh8KMMV/Hn6v8AMK4fTdK07U/iF4t+32Frd+X9j2efCsm3MRzjI4zgflXcVyXh/wD5KF4x/wC3L/0Uaiqk5RT7/ozpwUpQpV5RdnyL/wBLga//AAi3h7/oA6X/AOAcf+FH/CLeHv8AoA6X/wCAcf8AhWtRV8kexzfWq/8AO/vZk/8ACLeHv+gDpf8A4Bx/4Uf8It4e/wCgDpf/AIBx/wCFQ654VsfEOqaNfX812RpNwbqG3jl2xSS8bXcdSVIyvI6kHIJFblHJHsH1qv8Azv72ZP8Awi3h7/oA6X/4Bx/4Uf8ACLeHv+gDpf8A4Bx/4VrUUckewfWq/wDO/vZwOsa58MNA1xNF1RNEtr9tmY2sQQm77u9ghVOx+YjAIJ4OaseNPCEk3he6HhLS9Et9XXDxmbT4m3gHJRSw2qx6AsCO3Gdy+DfEP4Z+Kf8AhZWpjTtHvr+31G7NzBcwwHy/3rbirMCVTaxKksRwNxwCK+m/DekN4f8ADWm6O95JeGyt0g890VCwUYHC8AAcDqcAZJOSTkj2D61X/nf3s+KdSi8QaNcLb6omp2M7IHWO6EkTFckZAbBxkEZ9jX0R8C9Lg1j4fPc63pVtdOL2VLae6tEZniAU8OVy4Dlxkk4wR2wOC/aO/wCSh6f/ANgqP/0bLXr/AMEv+SQ6F/28f+lElHJHsH1qv/O/vZ1X/CLeHv8AoA6X/wCAcf8AhR/wi3h7/oA6X/4Bx/4VrUUckewfWq/87+9nyT428AeP9EvNQ1S6si1i0rStLpUhaCIEFztTO9I1GRllAGOvQmT4LapLc/EO00e/gttQs9QSRZEvYRMUKRu6shblTlcHsQeRkAj6S8d/8k88S/8AYKuv/RTV4B+zj/yUPUP+wVJ/6Nio5I9g+tV/5397PRfip4c1+2t9Pn8C6Bp5SNJ3vhHYWrsQAhTCyKSx+/wgJP5V836l4k1bVbhZ7i5VHVAgFrCluuMk8rGqgnnrjPT0FfdVfEHjv/kofiX/ALCt1/6Najkj2D61X/nf3s+yf+EW8Pf9AHS//AOP/Cj/AIRbw9/0AdL/APAOP/Ctaijkj2D61X/nf3syf+EW8Pf9AHS//AOP/Cj/AIRbw9/0AdL/APAOP/Ctaijkj2D61X/nf3syf+EW8Pf9AHS//AOP/Cj/AIRbw9/0AdL/APAOP/Ctaijkj2D61X/nf3syf+EW8Pf9AHS//AOP/Cj/AIRbw9/0AdL/APAOP/Ctaijkj2D61X/nf3syf+EW8Pf9AHS//AOP/Cj/AIRbw9/0AdL/APAOP/Ctaijkj2D61X/nf3syf+EW8Pf9AHS//AOP/Cj/AIRbw9/0AdL/APAOP/Ctaijkj2D61X/nf3syf+EW8Pf9AHS//AOP/Cj/AIRbw9/0AdL/APAOP/Ctaijkj2D61X/nf3syf+EW8Pf9AHS//AOP/Cj/AIRbw9/0AdL/APAOP/Ctaijkj2D61X/nf3syf+EW8Pf9AHS//AOP/Cj/AIRbw9/0AdL/APAOP/Ctaijkj2D61X/nf3syf+EW8Pf9AHS//AOP/Cj/AIRbw9/0AdL/APAOP/Ctaijkj2D61X/nf3s+Yfi14l1Xw18QLrS9M07S7CwiiiMA/siBvODKCX3Ohz8xZeMD5MYyDnD8HeN9f1HxjpVnJpmn6tFNcKktlHo9qGkQ/ewQi4IXLZLADGW4zX0l8Q7Czu/APiGW5tIJpItKuvLeSMMU+TfwT0+ZEb6op6gV80fBL/kr2hf9vH/pPJRyR7B9ar/zv72fVP8Awi3h7/oA6X/4Bx/4Uf8ACLeHv+gDpf8A4Bx/4VrUUckewfWq/wDO/vZk/wDCLeHv+gDpf/gHH/hR/wAIt4e/6AOl/wDgHH/hWtXwR9vvP7R/tH7XP9u83z/tPmHzPMzu37uu7POeuaOSPYPrVf8Anf3s+4f+EW8Pf9AHS/8AwDj/AMKP+EW8Pf8AQB0v/wAA4/8ACtaijkj2D61X/nf3syf+EW8Pf9AHS/8AwDj/AMKP+EW8Pf8AQB0v/wAA4/8ACtaijkj2D61X/nf3syf+EW8Pf9AHS/8AwDj/AMKP+EW8Pf8AQB0v/wAA4/8ACtaijkj2D61X/nf3syf+EW8Pf9AHS/8AwDj/AMKP+EW8Pf8AQB0v/wAA4/8ACtaijkj2D61X/nf3s4H4h6Bo1l4F1K4tNIsIJ08rbJFbIjLmVAcEDPQkV31cl8TP+Se6p/2y/wDRqV1tZxSVR27L9Tqr1JzwVNzd/env6QCiiitjzjh7PWfGWs3mq/2Wmgra2d/LZr9qEwc7D1O0kdCPTvxVv/i4f/Ur/wDkxR4E/wCZl/7D11/7LXW1z04OUbts9fF4iNGs6cacbK3TyRyX/Fw/+pX/APJij/i4f/Ur/wDkxXW0VfsvNnN9e/6dw+45L/i4f/Ur/wDkxR/xcP8A6lf/AMmK62vjzx94y8TP8QdaUa/qUaWepzpbJFctGsIQtGuxVICnZkEjk5Oc5OT2Xmw+vf8ATuH3H0t/xcP/AKlf/wAmKP8Ai4f/AFK//kxWr4TvrjU/Buh395J5l1dafbzTPtA3O0aljgcDJJ6VsUey82H17/p3D7jkv+Lh/wDUr/8AkxR/xcP/AKlf/wAmK62ij2Xmw+vf9O4fccl/xcP/AKlf/wAmKP8Ai4f/AFK//kxXkvx2+IHiCw8UQ+H9Kur7S7W3iSZ5oS0L3DsDyrg5aMDjjHzB852jHX/Ajxdrvijw9qEWtNPdfYZY44L6UD94pTmPIUbmXaGLElj5gz2yey82H17/AKdw+46v/i4f/Ur/APkxR/xcP/qV/wDyYrraKPZebD69/wBO4fccl/xcP/qV/wDyYo/4uH/1K/8A5MV1tFHsvNh9e/6dw+45L/i4f/Ur/wDkxR/xcP8A6lf/AMmK62ij2Xmw+vf9O4fccl/xcP8A6lf/AMmKP+Lh/wDUr/8AkxXW0Uey82H17/p3D7jkv+Lh/wDUr/8AkxR/xcP/AKlf/wAmK62ij2Xmw+vf9O4fccl/xcP/AKlf/wAmKr399460zTrm/vJPC8draxPNM+25O1FBLHA5OAD0rta4f4wWNxqHwo1+G1j8yRYkmI3AYSORJHPPoqsffHHNHsvNh9e/6dw+48ys/wBoTV767S3i0+xDvnBaF8cDP/PT2r6Fr4T8P/8AIctv+Bf+gmvuylBNTcb30X6l4mUamFp1VFJuUloraJQa/NhRRRWx5x518QPiBe+EdSkiik0uK3jsBdf6Yrl5XLlRHGFPLHAwMdmJIAJHnH/DR2p/9A20/wC/Df8Ax2vR/E/hjS/F3xBm0jV4PNt5NByrLw8Tifh0PZhk/mQQQSD80+PvDUPhDxvqehW9xJcQWzoY5JAA210VwDjgkBsZ4zjOBnFYRi5Ntt7nq1q0aEacY04u8U9Vrc9+8C/EfxX8QPt/9k22iw/YfL8z7Wkq537sY2s39w9cdq7D/i4f/Ur/APkxXGfs4/8AJPNQ/wCwrJ/6Kir2Cq9l5sw+vf8ATuH3HBTaH4vuNcttblsPCD6nbRNDDdFJ96I3UA/nj03MBjc2dD/i4f8A1K//AJMV1tFHsvNh9e/6dw+45L/i4f8A1K//AJMUf8XD/wCpX/8AJiutoo9l5sPr3/TuH3HJf8XD/wCpX/8AJij/AIuH/wBSv/5MV1tY/inxJZ+EfDl3rl/HPJa2uzekCgudzqgwCQOrDvR7LzYfXv8Ap3D7jK/4uH/1K/8A5MVx/jf4leI/AX2aPVX8PTXVxylraLM8gTn5yCwAXIxyeTnAODjM+H3xu1jxd8QLfRLzS7GGxvPO8kwl/Mi2qzruYkh+FIOFXk54xg9R8VPhrpfi7TrrW3S+Or2Gnyi2S0bPnlQzohQqc/MT93BO4jJ4wey82H17/p3D7jzv/ho7U/8AoG2n/fhv/jtev/8AFw/+pX/8mK+OdJ02bWdZsdLt2jWe9uI7eNpCQoZ2CgnAJxk+hr7zo9l5sPr3/TuH3Hhfif43av4X1ybSJl0W9uIOJmso5XSN+6Es6/MO+M46ZyCB6N8OvFd34x8NPql5FDE/nlFWFSo27EYZyTz8x718neO/+Sh+Jf8AsK3X/o1q+kvgV/yTz/t5/wDaUVTZxmldmvtI1sNUbhFNWtZW3Z6bRRRW55gUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABXkU/8AZ3/DNl7/AGp/x7/ZJ9n3v9d5zeT93n/WbPb14zXrteNapY3Gofsy3cNrH5kixNMRuAwkd15jnn0VWPvjjmsn/FXo/wBDup/7jU/xw/KoeQ6N8PG174Uan4qsHka/069dZoWkVUNskSuzKMZLgtn7wGAcAnAMnwY8Sf8ACOfEqw3R74dS/wCJfJhcsvmMu0jkY+cJk8/Lu4zivX/2cf8Aknmof9hWT/0VFXgnjzwpN4M8Y3+jOJDAj77WR8/vIW5Q5wATj5SQMblYDpWpwn23Xz54ytJviZ8erbQIBHPpejJGLxTdHyzGGVpyNvKuS6xELzlRkjHy+r+EPGa658N7XxZqkcdmn2eWa68vcyoImZXYDBOPkLY5IzjJ6nk/gFo72/gu68QXb+dfa1dvK87Ss7uiEqN+f4t/mnIzncMnsAD588feGofCHjfU9Ct7iS4gtnQxySABtrorgHHBIDYzxnGcDOK+n/gxBNbfCTQUnikicpK4V1Kkq0zsp57FSCD3BBr58+Nv/JXtd/7d/wD0njr6T+GNnNY/DLw7DPdyXTtZJKJHzkK43qnJPCqwQeyjgdAAbHiPX7Hwt4fvNa1JpBaWqBn8tdzMSQqqB6liBzgc8kDmvly+m8Y/HLxbJJaWv+j22BHEZNsFjEzAZZj1Y4ySAWbacDCgDt/2kfEn/IJ8LpH/ANRCWRl/340CnP8A10yCP7uD1rrPgHoFjpvw8h1iBZPtmqu7XLs2RiOR0RVHQADJ9csecYAAPONU/Z98X6NcNe6DqVpfG3dHtjHI1tcFsjkA/KpU5Od/QZ68Vv8Awa+Lt9qOqL4b8UX8cplQLYXk52uzjCiJiBhiw5DMckgjLFgB75Xxp8WrOax+KniCGe7kuna4EokfOQrorqnJPCqwQeyjgdAAez/HW/8AG8WlvZ6NYyHw7Lbg313bKWlB+cujYOVi2qpLbQOxbBIPz54V/t3/AISjTv8AhGvP/tnzR9l8jG7djnOeNuM7t3y7c54zX2H8QZ4bb4c+JHnljiQ6ZcIGdgoLNGVUc9yxAA7kgV4J+zj/AMlD1D/sFSf+jYqAPT/B3i34k614ogsdb8JQadptvvj1C6Mbpl8SbDEWfDLkIp27+5yAwxn/ALR3/JPNP/7Csf8A6Klr2CvG/wBpCNj4F0yUTSBF1NVMQC7WJikwx4zkYIGCB8xyDxgA88+EXjn/AIRrTtT0fSPD8+o+JtUlQWTK+Y3wCAJBkbVTLvkdQSCVA3Vuarqnx80XS7jUr95EtLZN8rpFZSFV7nagJwOpOOBkngGtv9m7RrNPD2ra5s3X0t39j3sAdkaIj4U4yMl+ecHavHFeoeO/+SeeJf8AsFXX/opqAOT+FPxWXx8k+n6hbx22tW6GZlgVvKli3AblySVILKCCe4IJyQvplfJHwKuvs/xX02L7PBL9pini3ypuaLEbPuQ/wt8m3P8AdZh3r1P9orX20/wdY6LE0ivqlwWkwqlWiiwxUk8g72iIx/dPPYgGP4z+OGpanqiaJ8OoJLp3TP2xbRpJZGGGIiiYdAoIJZTnJwBgMcDxL4v+NfhC3guNdu5LWCdykcgt7SRdwGcEopAOMkA4zg46GqfwU8c+FvBl5qH9uRTw3V5tRL8RiRIo1BJUgDeuWxnG4H5chduT7HP8ZvhtdW8tvca3HNBKhSSOSwnZXUjBBBjwQRxigC58MviJb/EHQ5JTD9n1Oz2pewqDsBbO10J/hba3BORgg54Y+b/Ej4s+PfDHiiWxj0qDTLCK7zazSwGT7bEgGRvJ2lW3AkIAy5C5BBzxHwKvri0+K+mwwSbI7uKeGcbQd6CNpAOenzIp49PTNer/ALR3/JPNP/7Csf8A6KloA5jT/wBpG8Gh341LRoH1fj7E1sCsBzx+8DMWGOvyn5uny/ePHN8VvGHim9fTtS1KM6fdOzPax20aqoHzqobbvwCB1YnjkmrvwF8PaRr/AI1uxq9taXiW1k0kVrcIXDMWVd+3GwhQSMMerKQDjK+lfEP4W+GdPtLvxXp1vJYXcCRqLa12pbsSwQsU28Ha38JAyAcZJzlX/hS9Gd2V/wC/Uf8AHH80ew0UUVqcIVwWga3p3hzR/FmratcfZ7GDXrjzJdjPt3MijhQSeSBwK72vkv4r+IbhtRv/AA0q7bVNZur+Q5B8xyfLXtkbQr9+d/TgVlP44/M7sP8A7rW/7d/M2dS+O3jfxLfrY+GNOjsnkcNDFawG7uGAQ7lO4EMOrcICABzwc17/AFT426E//CQXj6yiXNuXc+UksUUaqpJeEApCQAMkqp+9/tV2/wCzfpVivhXU9YFtH/aD3rWpuDy3lKkbBB6DcxJx14znAx7ZWpwnlfwm+LMPjK3TR9YeOHxBEnBwFW8UDllHQOByyj/eHGQup8Svilp3gKza1jX7TrssQktrVkYIFYsBI7dNoKn5Qdx4HAO4fOnj6ybwX8WdTXS5Y4ns71Ly1McKqsJbbMihORhdwHodvQdK+v8AUtJ03WbdbfVNPtL6BXDrHdQrKobBGQGBGcEjPuaAPiDw1qUOjeKtI1S4WRoLK9huJFjALFUcMQMkDOB6ivoHU/j74V1nSbzS7fT9ZWe9ge3jaSGIKGdSoJxITjJ9DXg3guCG68deHre4ijmgl1O2SSORQyuplUEEHggjjFfVPij4f+EI9EvdSh8OabBd2VncPA8ECxBWMZ5KrgMRgEFgdp5GDUz+Fm+F/jw9V+Z0nhb/AJFDRf8Arwg/9FrWtWT4W/5FDRf+vCD/ANFrWtRD4UGK/jz9X+YV87/E7x3r/gn4jam2h3UcIu0RZleFXDEQKEbkZBUuSOxOMgjivoivKvEfhhPF1z8RNL8jzroxWctmBtDCdYWKYZuFyflJ4+VmGRmoqfFH1/Rm+E/g1/8AAv8A0uBynwv+LnjXxP4o07QLq1sb+E73urswtHIsYBO4lPkGCVUfIATtBIJ3V6Z8UfF+peCPBx1fS9Pju5/tCRMZVYxwq2fncLg4yAo5HLjnsfmD4beKF8IePNN1WeSRbPeYbva7AeU42ksACWCkh9uDkoO+DX2Xf31vpmnXN/eSeXa2sTzTPtJ2ooJY4HJwAelanCfOFr+0j4kTz/tmjaVLmJhD5Ikj2Sfws2Wbco5yo2k/3hX0H4c1K+1fw/Z6hqWlSaVdzoXeykk3tEMnbk4HJXBwQCM4PIr4csL640zUba/s5PLurWVJoX2g7XUgqcHg4IHWvu+wvrfU9Otr+zk8y1uokmhfaRuRgCpweRkEdaAPBNf/AGh9esXayi8Ix6XqEbgypqMruQpXONm2MgnKkEnp25yO3174rTeGvhz4e8SXmiST3eqJHvtw5iRcxliwcBwAcAqpO4g+qsBhnR08e/tBX91O+bDwvFbCGa0lXmdXEipJnOfmMwO0DGwAkHr7RQB4PYftKW9xqNtDeeG/slrJKiTXH24yeUhIDPtEWWwMnA64r3ivjiwsbfTPjrbWFnH5dra+JUhhTcTtRbkBRk8nAA619j0AfNH7SNjcR+MtJv2jxazaf5Mb7h8zpI5YY68CRPz9jXq/wS/5JDoX/bx/6USV5B+0d/yUPT/+wVH/AOjZa9v+Ff8AaP8Awq/w9/an/Hx9kGz7v+pyfJ+7x/q9nv685oAseLPiF4a8FbY9Zv8Ay7qSJpYrWKNpJJAPYDC5PALEAkHng48n/wCF3ePNe/0rwv4I82xT925+zz3n7wcn549gHBX5cZ755rzDxTfP8QPirdyWUkH/ABM9QS1tJCrIhTKxRMwOWGVCk8dzwOlfX+h6NZ+HtDstIsE2WtpEsSZABbHVmwACxOSTjkkmgDyPUvjWX0bXNL17Q9S8K6pJpkrac0pkLSSFWVcfIrId2MNjHB5GBnzz4A6n9g+KEFt5Pmf2haTW27djy8AS7sY5/wBVjHH3s9sH6T8XeF7Hxj4au9Hv44yJUJhlZNxglwdsi8g5BPTIyMg8E18+fs4/8lD1D/sFSf8Ao2KgD6fr4w13TP7b+Mmp6T53k/bvEEtt5u3ds33BXdjIzjOcZFfZ9fIH/Nwv/c1/+3dAH0X8RviHD8PdLtruXSru/e5cpF5ZCRKwwcPJztJUsVABztPTGa87/wCGl7P+zt//AAjE/wBu83Hk/bB5fl4+9v2Z3Z427cY53dq9wv7G31PTrmwvI/MtbqJ4Zk3EbkYEMMjkZBPSvhyeO+8J+KpYkmjXUNJvSqyxjcolif7w3DkblzyPqKAPuOwuvt2nW159nnt/PiSXybhNkke4A7XXswzgjsa8L1b9pCa1uL6zt/CUkM8TyRRteXRVkYEgeZEEyCD1Xd6jPevdLC+t9T062v7OTzLW6iSaF9pG5GAKnB5GQR1r5w+MWk2/iz4q6daeHbmxuL66xp9zHBkvFPGcs8u1T8oR1BbnHlODjYRQB6v8Mvib/wALG/tT/iUf2f8AYPK/5efN379/+wuMbPfrXoFY/hjwxpfhHQ4dI0iDyrePlmbl5XPV3PdjgfkAAAABsUAFeR+LfjxpvhfxjLoaaTJfwWzql1dRXKjY38YRcEMVBwQWX5gQcYzXrleV/HXwjY6x4IutcWykfVtMRWilhXLGLeN6vgcoqln/ANnBOQC2QA8FfG/TfGPiwaCuj3do87uLOYyK4dVV3JkHGw7VHA38nrxk9B8QfiTpvw8t7Rr2zu7qe9SU2yQ7QpZAvDsTlQS45Abvx6+Ifs8alaWfxBuLSdYxPe2Tx28hDltylXKDB2gFVZiWH8AAIyQfd/iSugL4D1K68SWEd9YWqCZYWZlLS5xGFdfmQlmC7h0DHPGaAOL/AOGjvB//AEDdc/78Q/8Ax2vUP7b07/hHv7f+0f8AEs+yfbfP2N/qdm/dtxu+7zjGfavjjwLc6XonjTRr/wAT2Pm6M+8us1t5qOjB4w+0j5lV+uM/cOASMV9pwQQ2tvFb28UcMESBI441CqigYAAHAAHGKAPI/wDho7wf/wBA3XP+/EP/AMdr1SDVbG50aLWEuYxp8luLpbiT5FERXdvO7G0beecY7180aN8P9N+JPxS8Rrpfl6Z4dsbjJazZZVf59oEfICiQLI4IDKvAwRiun/aK1VtN0vQfDmnXMdtZyJI89jBtUbE2CHKjkIDvwOmV9VGADU8Q/tGaFp94IND0yfV4x9+d5DbIeARtBUsepByFxjjIOaxIP2mZlt4luPCkck4QCR478orNjkhTGSBntk49TU/7OXhexkstQ8UTxxy3iXBs7bcnMACBnZTnGWEgHTICnnDEV7RqvhvRdcS4XU9LtLk3Fv8AZZZHiG9ot27Zv+8AGwwweCARgjNAGX4L8f6F48s559HknElvt+0QTxFHi3FguSMqc7CflJ98HiqfjX4o+HPAdxDaao13NeSosi21rDuYRksN5LFVxlCMZz04xzXy5Z31x8OviU01rJ9pk0bUJISdoTz0Vmjcc7tu5dw74zxyK+k/it4E0DxB4a1fXb61k/tSx0yRre5SZlKiMNIF252kE5ByM4JwRwQAR+Hvjj4K1+8Nq91Ppcn8B1JViR+CT84ZlXGP4iM5AGTRrHxx8FaPriaYbqe8U7PMvLJVlgj3erBstgYJ2hvTkggfMHhXw9ceK/FGnaHatsku5QhfAPloBl3wSM7VDHGecYHNeySfszTB4RF4rjZC+JS1gVKrtPKjzDuO7aMHHBJzxggHsfjv/knniX/sFXX/AKKavmD4Jf8AJXtC/wC3j/0nkr6T8S6bDo3wj1fS7dpGgstCmt42kILFUgKgnAAzgegrwj9nWeGH4jXSSyxo82mSpErMAXbzI2wvqdqscDsCe1AH0fr/AIj0jwtpbalrV9HaWgcJvYFizHoFVQSx6nAB4BPQGuT/AOF2/Dz/AKGH/wAkrj/43XnnxJ8SzfFfxBaeBfCFvHeQW9x58+oEny9ygqWDDgRKHOW53EgLnjf1+gfAPwfpulrBrEEmsXm8s1y0kkAx2VUR8AAepJyTzjAAB1mvfEXwj4a+x/2rrcEX2yLz7fylebfGej/uw3ynsTwcHGcGuf0n4gfDHW/GkRsWsW1ufaIL+SwMbyuQybBKyhg20BecZ3KoJOQJPiP8NtA8QeGr68TSpP7UsdMaLT/shZSojDNHEsY+Ugn5cbc4OBjjHgnwS/5K9oX/AG8f+k8lAH1Xr/iPSPC2ltqWtX0dpaBwm9gWLMegVVBLHqcAHgE9Aa4+T42+CE8Sw6MuoSSB38tr9UH2WN8kYZyQcZA+YArhgd2MkdJ4x8HaX440P+ydW89YVlWaOSB9rxuMjIyCDwWHIPX1wR8UX9jcaZqNzYXkfl3VrK8MybgdrqSGGRwcEHpQB9h6/wDFbwb4f0tr5tatNQO8Itvp08c8rE+wbAAGSSSBxjqQDH4O+LHhbxtefYbCee2vzuKWl5GEeRVAJKkEqep4zu+VjjAzXL+GP2fPDWn2cMniBp9UvjFieNZmjgVyc5TbtfgcZLc8nAyAPKPi/wCGdO8CePrSLw4J7KNrSK8jCzMTDIHdcoxO4fcDckkEnBxgAA+t6z9b1vTvDmjz6tq1x9nsYNvmS7GfbuYKOFBJ5IHArH+HXiG48VfD/R9YvFxdTRFJjkfO6M0bPwABuKlsAcZx2rwj9oHxZeal4v8A+Ea2eVY6VtfCuT58kkatuYdPlDbRxkZbn5sAA7//AIaO8H/9A3XP+/EP/wAdrY/4Xr4B/tH7N/ak/k+V5n2v7JJ5e7ONmNu/djn7u3HfPFV/DfwJ8I6Ro8lrq1v/AGxdzZEl1KXiwu4EBFVvkxtHIO45YZ2nbXhHxY8HW/gnxzNYWPFhcRLdWqFy7RoxIKkkdmVsdfl25JOaAOz1n43Xni651DQ49Lgh0i7wLZ2JE6bGVwznJU52H5QBjcPmOPm+lK+X/DXw/wBBm+FKeNYr67fVoXZJYFmQxI3neXhl27gdjK2C3cHoa+oKyX8V+i/U7qn+40/8c/yphRRRWpwnl2lePvC/hC416313Vo7WefW7t44xG8jbQVGSEUkDOQCcZwcdDWrB8Z/h9c3EUCeIow8jhFMltMigk45ZkAUe5IA715P48+Hdxr1n4l8V6fNuuNN1O7S5tnIAaBDvLqf7y7mJB6jpyMN5BpMdjNrNjFqk0kGnvcRrdSxjLJEWG9hweQuT0P0NZUfgR3Zl/vUvl+SPvOvP/wDhdvw8/wChh/8AJK4/+N16BXyB4+8EadpfxVXwt4dusrdywxrHOGAtZJSMRlsEsoDK27k4bByQSdThPqex8VaFqHhyPxDDqcCaQ+cXc5MKDDlOd+MfMMc9a8/1HR/grr+sX1/e3mhvePLmdxqphV3KglgFkVWzkZZerbsndmjVvghb3/gbR/DVt4ivrddPleZ5JFMsczuPmPlbgEwfu4PAZ87ixavmC/sbjTNRubC8j8u6tZXhmTcDtdSQwyODgg9KAPvOCCG1t4re3ijhgiQJHHGoVUUDAAA4AA4xXP8AiXx94X8IXEFvrurR2s86F44xG8jbQcZIRSQM5AJxnBx0NdJXw5pMbeLPHVjFqk0jPq2pxrdSxhVYmWUb2HGAfmJ6Y9qAPqP/AIXb8PP+hh/8krj/AON11mgeI9I8U6WupaLfR3doXKb1BUqw6hlYAqehwQOCD0Iryf4g/ArQ28NTXnhOzks9Qs0aX7OJJZhdKBkoAxYh+Plx1JwRyCsnwL0b+0/hBrGnXqTw2up3dxEJFG0tG0KRsyEjBwQwzyMqfSgDpPGlj8MPFdw0XiLVtGXULZHthKNSSKeA5OQfm5KtkgOCAc8cnPWeGNE0LQdDhtfDlvBDpsn7+MwuZBLu537ySXyMYJJ4AA4Ar5I+I/gX/hX/AIht9J/tH7f51otz5vkeVjLuu3G5v7mc5719V/D6Novhz4bV5pJidMt23OFBAMYIX5QBgA4HfAGSTkkA6SuP1n4p+CdA1F7DUdfgS6TIdIY5JthBIKsY1YKwIOVPI9K8c+PXxBbUtUPhLTJ5Fs7J/wDTmSRSlxLwQnHOEOcgn72cjKA1oeCPgBpur+GrLVtd1e7L31vHcRQ2O1BErDcAzOrbiVK9AMHI+bg0Aez6V4y8M649vFpmv6bcz3Cb4oEuV81ht3f6vO4EDJIIyMHOMVuV8mfEP4WX3wzt9M1i01iS7R7jZ9oji+ztbygbo8Yckk7XORjGz3Feh/Cb4gap450PU/B+oXU8WrR6fIbbV0OXCHCBm5BMil1IYH5sckEZYA9wor4o+IHhTVPB/i25sNWuftk02bmO7LZNwjMf3jZJIYkNkHuDyRgnY+Hvg/xB8Sd+iLrU9toWnYlkEsjSRwu2/bsh3AFid/PGBuOckAgH1XpviXQdZuGt9L1vTb6dULtHa3SSsFyBkhSTjJAz7itSvljx/wDBG48E+F31yDWf7SjhlRJ0+yiHy0Y4D5Mhz8xUYA/iz0Bru/gF481LX01HQda1GS8uLZFuLV59zytGWIkDOeoVimM8/MeoACgHtlcPffGDwDp95JazeI4HkTGTBFJMhyAeHRSp69jx061x/wC0P4n1TR9D0zSLCfyLfVfPF0ycO6Js+QHsp3nPrjGcEg8B8Jfhn4b8d6dey6trM8V9FKVjsbSWNJBGoXMhDBiVJcLkAAEHk54APoOD4g+Dbm3inTxVowSRA6iS9jRgCM8qxBU+xAI71qa7pn9t+HtT0nzvJ+3Wktt5u3ds3oV3YyM4znGRXyo/hqHwh8etN0K3uJLiC21iyMckgAba7RuAccEgNjPGcZwM4r67oA+E/D//ACHLb/gX/oJr7sr4v1OFbb4raxAhkKR6ndopkkZ2IDOOWYksfckk96+0KyX8V+i/U7qn+40/8c/yphRRRWpwnJf81e/7gP8A7Xr59+P39nf8LQn+xf8AHx9kh+3fe/12Djrx/q/K+7x+Oa+gv+avf9wH/wBr182/G3/kr2u/9u//AKTx1lS+16ndjv8Al3/gR6/+zj/yTzUP+wrJ/wCioq9A/wCE78H/APQ16H/4MYf/AIqvkzwP4V8TeM7i80bQZpIrSRFe+aSVkt8KSU8zbncd2dowTnJ6AkdR45+ButeEtLk1WxvI9XsIEDXBSExyxDnLbMsCgAGSGyMkkYBNanCfVdFeN/BD4mX3ilJfDmsCSe/srfzor0nJliDKuJO5cFl+b+IdeRluw+Kni2bwZ4DvNRs5o4tQldLezLxlx5jHk46ZCB2G7jKjOehAOo1LVtN0a3W41TULSxgZwiyXUyxKWwTgFiBnAJx7GjTdW03WbdrjS9QtL6BXKNJazLKobAOCVJGcEHHuK+XPAvww1j4pxXevX/iHy41lNu884e5nkkVUIyCR8u1gM7s8YxjmsvxX4Q8R/CLxLp95FqEYdneSxvbZsE7TghkPQ7WXKnKkPjLc0AfYdZfiPQLHxT4fvNF1JZDaXSBX8ttrKQQysD6hgDzkccgjiuf+F3jWbx54OGqXcEcN5FcPb3CxIVjLDDAplmONrpnJ657YrzD4+2/i/S78arDrt2PDt8gs/skM7RLE2w5R0BxIHAc7jnupwAuQDu/DHwS8LeFdch1e1uNVnuoOYfOugoRvX92qk8ZBBJUhiCDXpFfMH7OP/JQ9Q/7BUn/o2Kvp+gD4EgnmtbiK4t5ZIZ4nDxyRsVZGByCCOQQec19918KeGtSh0bxVpGqXCyNBZXsNxIsYBYqjhiBkgZwPUV910AfFnxO02bSvib4it52jZ3vXuAUJI2ynzVHIHO1wD7569a+hvgV/yTz/ALef/aUVeW/tHf8AJQ9P/wCwVH/6Nlr1L4Ff8k8/7ef/AGlFWU/jj8zuw/8Autb/ALd/M9NooorU4QooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACvE/EH/Jr9z/AMB/9LRXtlfH/iKbxve2mm6DGNSm0e7SR9Ps7WMlbgK26TIQZcq6lsNkrwRgEE5P+KvR/od1P/can+OH5VD2D9nH/knmof8AYVk/9FRVl/tIaBYnRtM8RhZBqC3C2JYN8rRFZHAI9QwOCMfeOc8Y8o0CT4h+CNUWLR7HWdPvNRQosDWDMbgJ8x2xuhDFQc5AyAT0BNfUet6JeeLvhrPpOrW8EWp3unr5kW8iOK62hhypJ2rIAeC3A799ThPjzSoda1x7fw5phu7kXFx5kVikh2NLtxv252ghc5Y9ADkgCvt/SdNh0bRrHS7dpGgsreO3jaQgsVRQoJwAM4HoK8X+AvgPV/D+s65qmt6dd2M8aCxgWXAWT5t0hHdgCkeGB2nJwT27T4s+LfEHhXw9FJ4c0ye5upvMM10tq00dpEqfM7EHCtkqRuBXCtkcUAfOnxa1KHVfip4guIFkVEuBbkOADuiRYmPBPG5CR7Y6dK+o/hxfW+ofDXw5NayeZGunwwk7SMPGojcc+jKw98ccV8iQ+GvFWuodUg0TWdQS5dnN2lrLKJW3Hcd4B3HdnJz1zXs/wY8XeMoNbh8F6vpN3LYWqFXmuIJFlsAULxq5IwEIUhQwB+YYOAFoAj/aU0T/AJAevxW/9+yuJ9//AAOJduf+uxyB9T0rqP2fvENvqfw//sdV2XWkyskgyTuSRmkV+mBkl1xk/czxkV6J4j0Cx8U+H7zRdSWQ2l0gV/LbaykEMrA+oYA85HHII4r5s/4Q/wCJPwr8UedoNtfX1qZcrJZRPLDdoo4E0SEkcORhuh3bScbqAPqevjj4iomv/GHWLXRbPZJcagLSOHCx75xtjY9cfNIGOSed2Tg5ruNZ/aR1S7054dI0KDT7p8j7RNcfaNgIIyq7VG4HBBORxyDmuz+C/wAMJvCVv/wkGqSSLql7b7Ft1JCxQuI3w6sgYShlIIyQKAOk+ME1vB8KNfe6tftMZiRAnmFMO0iBHyP7rFWx324PBrxz9m+CZvHWp3CxSGBNMZHkCnarNLGVBPQEhWIHfafSvoPxV4et/FfhfUdDum2R3cRQPgny3Byj4BGdrBTjPOMHivkSCTxX8KPGMUrwyafqcSBmikIeOeJv4TtOHQ4xweCvBDLwAfadeH/tKaZ5vh7Q9W87H2a7e28rb97zU3bs54x5OMY53dscx+BvHXxQ8Y+JdO1M6RH/AMI2HWK6WCJYInViymRXlJdyh5IQ/wAAGATz558SvG/jjxFp1jYeKdF/sm1WVpok+xSwec4GM5kJJ2hj0x9/nPGAD1v9nWRX+HN0qwxxlNTlVmUtmQ+XGdzZJGcEDjAwo4zkn0DxpPNa+BfENxbyyQzxaZcvHJGxVkYRMQQRyCDzmvEPg/qnjLwbcf2Fd+DNZl0/Ub2I+dJbSQraFiEkkJMZ3DbsOCQBs9ya0/jd8T9U0q8u/B+n2P2eG4tNtzd3EeTKkg5EPbbjcpYg87gNpXJAOE+A2mzX3xUs7iJowlhbzXEoYnJUoYsLx13SKeccA/Q93+0vY3EmneHb9Y82sMs8Mj7h8ruEKjHXkRv+XuK8c8DeM77wL4lj1ixjjmBQw3ED8CaIkErnGVOVBBHQgZBGQfqPxh4et/in8NYUtW+zyXcUV/YPcAjy3K5XeFPdWZT97G7IBIFAHN/s+vDffDkCWytA9hqcoikWIbyxjU72P9/bIyZGPlAH17z/AIQTwf8A9Cpof/guh/8Aia+XPA/jfWvhT4gvIrjS5Ck6Kt5p90pgkyATGwJUlSN2ehBDHjoR6fJ+0tpo0uGSLw3dtqBfEsDXKrEq88rJglj93goOp545APWJNU8OaX4sh00vaQa9rCb9kcX724WJTguyjoFDBSx7EDoa87/aO/5J5p//AGFY/wD0VLWf8H9B13xTri/EjxTe/a5DE9vpwJAPGUZwqYVFH7xQuOSzNgHBbkPjZ8Sv+EkvJ/Cttp/k2ul6g/mXEj5eWSMNHwo4Vcl+5J+U/LyKALH7N32P/hMtW3+f9u/s/wDdbceX5fmJv3d92fLxjjG7PavaviZ/yT3VP+2X/o1K+UfBXjXUvAeszappcFpNPLbtbst0jMoUsrZG1lOcoO/rX094v1P+2/g4dW8nyft1pa3Plbt2ze8bbc4GcZxnArKv/Cl6M7sr/wB+o/44/mjvaKKK1OEK+N/irBMvjrU7hopBA93cIkhU7WZZWLAHoSAykjtuHrX2RXi2v+Bm8ceDvFEFnFG2rWfiC6nsyxVdx+UNHuI4DL2yBuVMkAVlP44/M7sP/utb/t38yz+zj/yTzUP+wrJ/6Kir2CvlT4U/Ei+8A6pP4f1DTLu5s7i4Ia0gh/0qK54TCqcFiSqqUPOQCMEEN6n4h/aB8I6ZZhtH8/WbpukaRvAi4I+8zrkZBONqt0wcZzWpwngnxO1KbVfib4iuJ1jV0vXtwEBA2xHylPJPO1AT756dK+06+ePgr4HuPEmuS/EHXpPMxdySW8bwgC4nOS0xyNu1WY42/wAa9tmD2/j/AOMdn4U1h/D2m2X2zW/kUm4kENtAzrlS7sRnGUJHC4b74INAHzh4E/5KH4a/7Ctr/wCjVr7J8U/8ihrX/XhP/wCi2r4m0LU/7E8Q6Zq3k+d9hu4rnyt23fscNtzg4zjGcGvpnRPixpfxA8K+ILKOznsdTh0u4mkt3PmIUwVyrgDOMpnIX73GcE1M/hZvhf48PVfmegeFv+RQ0X/rwg/9FrWtWT4W/wCRQ0X/AK8IP/Ra1rUQ+FBiv48/V/mFcl4f/wCSheMf+3L/ANFGutrkvD//ACULxj/25f8Aoo1FT4o+v6M3wn8Gv/gX/pcD51+OPh630D4lXL2rfu9SiF+Uwfkd2YPySc5ZGbtjdgDArpLr4gTeLfhRofgzw19ri8Qu8OnXFlEpJnt0iILCTGAjYUtkggbgcrkn0P47eGLfWvh/PqnkTyX+k/vbcw5OEZkEu5e6hRuJ7bM5AznzD9nbRPt3jm71aS33w6baHZLvx5c0h2rxnnKCUdCB9cVqcJ0fxV+EGkaV4Isrzw3ZyLeae8VvIAxeS8WR9oJAHzS+Y64xjgkdAoHOeH/i9b6F8F5/Ddqk8Guxb4LaVclSkrszShgQUZQzAdfm2EZG4L7/AOOPDf8Awl3gvVNDEnlyXUX7py2AJFIdNxwfl3KucDOM45r5o+D+jv4w+Iqw6o/22xTfqV9BdSswuXXKozD+Ng8oPzdQXznJBAPf/hZ4FTwL4Sjt5RnU7zbPesVXKuVH7oFc5VOQOTkliMbsDuKK4fxv8VfD/gLUbaw1OK+nup4vOCWkStsTJAJLMo5Ibpn7pzjjIB84WF9b6n8dba/s5PMtbrxKk0L7SNyNcgqcHkZBHWvsevhiw1a4uPGVtrF5qX2S6k1BLqa/8gSeU5kDNL5YGGwcttA5xivrOw+KPhzUPAd54wja7TT7NzHNG8P71ZMqAmASCW3pgg4+YZIwcAHin7R3/JQ9P/7BUf8A6Nlr2P4MTzXPwk0F55ZJXCSoGdixCrM6qOewUAAdgAK+dPir43s/Hvi2PU7C1nt7WG0S2QTkb3wzMWIBIHLkYyeme+B7f8BPE+l6h4EtvD8U+3U9N8wzQPwWR5WYOv8AeX5wD6HqOVJAPDIZ7Twj8Zy8Esljp+m66yFkZyY7dZirDjLMNmQRySMjnNfZdfMnx1+H19pniC68WWcEb6Teupn8mPb9mlwFJcDqHYbt/wDeYg8kFur+H/x50Y6Hbad4rlntb61iEZvSjzJcBcBSxG5/MIznIIOCcjO0AHqHjv8A5J54l/7BV1/6KavAP2cf+Sh6h/2CpP8A0bFW/wCPvjP/AMJNoeo6H4M06+uFaKUX949tkLajhnVRkhWB5Zwu0Hpk5XgPhb8QbPwN4ha81LTftNu9o1r5tsoWaNd+/pkLJlsAlvmAC4YBdpAPr+vjDXdM/tv4yanpPneT9u8QS23m7d2zfcFd2MjOM5xkV9d6p4j0jRvD7a9e30a6WqI/2mMGVSrkBSNgJIJYcjPWviC71W+vb2/vJ7mQz6g7PdsvyCYs4c7gMDG4BsYxkD0FAH3nXzp+0V4Shs7+x8VWkMgN65t71zICvmKg8ohTyCVVwccfIOhPPv8ApWq2OuaXb6nplzHc2dwm+KVOjD+YIOQQeQQQcEVn+MPDUPjDwnf6DPcSW6XaKBMgBKMrB1OD1G5RkcZGeR1oA8r+FHj/AE7wx8L9QtvEMn2SbQLt4DbeUyzN5hZ1TDdZC4mGOMBMtgAmqf7PGgX017rHi/UVuyblPs8FxM2Rcln3TMc/MxDIg3dMlhyQceEaTps2s6zY6XbtGs97cR28bSEhQzsFBOATjJ9DX3XYWNvpmnW1hZx+Xa2sSQwpuJ2ooAUZPJwAOtAFPxLDfXPhXV4NLMg1CSymS1McmxhKUITDZG07sc5GK+NP+EE8Yf8AQqa5/wCC6b/4mvt+igD4s0rwb4+tdUt5dM0DxBZ3m/ZFOltLAULfL/rMAKMEgkkDBOeK+06KKAPijW7G4+H3xKnhgjzJpGoLNai4YPvRWEkRfbjOV2k4x17Gvb/in47s9e8F6LouhQwahdeKfL8uCWQZiUkBdxWQbJBLtAzlcxuD0IrH/aU0T/kB6/Fb/wB+yuJ9/wDwOJduf+uxyB9T0qp+zl4Ymm1TUPFEjyJb26GzhVXIEkjYZ92G5Crt4YEEuCOUoA4v4pfDX/hXd5poi1D7Za30R2s6bXWRAokyOm0lgRzkZwc43N6vqXxPt4PgDaX9rqv/ABO7m0XT082UtOZ1CpM+VbcGCkyBiR95CeWAOh+0DodxqvgGC7s7D7TNp92JpZEQF4oNjByO+3OwnH93J4XI8c+Dngy+8UeNbW/gkjhs9HuIbu5lbkkhtyIq5ySxQ89AAT1wCAe//Cr4fw+BPDQEvmHVr9I5b4swwjAHEagEjC7mGRncSTnGAPHP2jv+Sh6f/wBgqP8A9Gy19P14n+0H4Jm1XS7XxRp9vJLcWCGK8VAWP2flg/XgIxbOAThyScLQBsfs+x2KfDINaTSSTveyteKw4jlwoCrwOPLEZ78seew9Ur54/Z28Y29pPd+Ebr5JLuU3VmyoTvcJ+8Vjnj5UUjjs2TnaK9f8deOtL8B6Gb+/Pm3EmVtbRGw9w47D0UZGW7Z7kgEA+WPipdWd58UPEMtjb+RCt2YmTYFzIgCSNgf3nVmz1OcnkmvrvxKyr4V1dnv5NPQWUxa9jVma3Gw/vAF5JXrgc8cV8wfCvwpd/ET4gy6tqgjuLO2uPtupM+webI5ZlTZgghmByMAbQwyCRn3/AOJniTRdJ8Fa9Y3+qWkF5caZMkNs0o82QurIu1PvEFuM4wMEnABoA8E+AOmfb/ihBc+d5f8AZ9pNc7dufMyBFtznj/W5zz93HfI+r6+RPgr4osfC3xBjm1KSOG0vbd7N7iR9qwlirKx4PG5AvOAN2ScCvreCeG6t4ri3ljmglQPHJGwZXUjIII4II5zQBy/xO1KHSvhl4iuJ1kZHsntwEAJ3SjylPJHG5wT7Z69K+ONN0nUtZuGt9L0+7vp1Qu0drC0rBcgZIUE4yQM+4r6v+NV/pEPw31HTtS1OOynvk/0RShdppI2WQKFHIBKhSx4XcM9gfBPgvriaD8TLGW5v4LKxnimhupZ3VE2bCwBZvu/Oidx6d8UAbnwN+IGm+E9Uu9J1jy4LTUnQpeFVAhkGQBI2M7DnqThTzgBmYfUdfNnx6+Hzabqh8W6ZBI1nev8A6cqRqEt5eAH45w5zkkfezk5cCt/4MfFXSIfDUPhzxFqkdpd2jlLSa5JCPDgsAZCSFK4KgHaMbAMnNAHsHiXTZtZ8K6vpdu0az3tlNbxtISFDOhUE4BOMn0NfKnwS/wCSvaF/28f+k8lfSfxH1bTdP8C65b3uoWltPd6ZdJbRzTKjTN5RGEBOWOWAwPUetfLHw01+x8L/ABD0jWNTaRbOB5FldF3FA8bJux1IBYE4ycA4BPFAH2nXxppOpQ6z8bLHVLdZFgvfEcdxGsgAYK9yGAOCRnB9TX1/quq2Oh6XcanqdzHbWdum+WV+ij+ZJOAAOSSAMk18SWGvfY/GVt4h+xQL5OoJffZLceVGMSB/LTrtXjA64HrQB9z182ftJ3kL+KtGsltI1nhsjK9yMbpFdyFQ8ZwpjYjk/fPA7/R8E8N1bxXFvLHNBKgeOSNgyupGQQRwQRzmvlj4+6tpus+OrG40vULS+gXTI0aS1mWVQ3mynBKkjOCDj3FAHufwfmt5/hRoD2tr9mjEToU8wvl1kcO+T/eYM2O27A4FeGfH/RJtO+JD6i3mNBqlvHKjGMqqsiiNkDdGICKx6Y3jjufc/g/a/Y/hRoEX2iCfdE8u+B9yjfI77Sf7y7trDswI7V5/+0j4ht107SfDSruunl+3yHJHloA8a9sHcWfvxs6cigD2jQ9Zs/EOh2Wr2D77W7iWVMkErnqrYJAYHIIzwQRXyp8XtfXxl8TZV0lo76CBI7Gza1VmM5zkgf3j5juAV4IAxnqfQ9G/Zx+z7DqfimfyZogl5bWMHl7+jbRIzHKh1U8pztHAPTuPCPwd8LeDdYXVrI311fR58mW7nB8rKsrYCBQchiPmB6DGKAMqfw9ceFf2fY9HvGzdQxRvMMD5He4EjJwSDtLFcg84z3r1WuS+Jn/JPdU/7Zf+jUrrayX8V+i/U7qn+40/8c/yphRRRWpwnJeBP+Zl/wCw9df+y18o+PPCk3gzxjf6M4kMCPvtZHz+8hblDnABOPlJAxuVgOlfV3gT/mZf+w9df+y15x+0H4GvtTS18V6dFJOLK3MF7GpyUiDFlkC4yQCz7jngbTjAYjKj8CO7Mv8AepfL8kdB4I+Iaw/A8+JNVvJL+70tJILgybgzyhsRIW2nJZXiG/B+9knINc/8CfCl3fXt/wDEHWhHLcX7yfZHOzJZnbzpdoHyEsCowRwXGMEZ8Y8OR6/4pFn4G02aM291em7SGQKqrKIyGkL43YCA8DPTgE19j+HNAsfC3h+z0XTVkFpaoVTzG3MxJLMxPqWJPGBzwAOK1OE1K+IPHf8AyUPxL/2Fbr/0a1fb9fBk7Lpusytpd/JKltcE2t7GrQswVvkkA6oTgHHUUAfedfInxB+EmteDtUmewtbvUdFKNNHdxRFzCi8sJtowpUfxcAjkY5C/Veu3V5Y+HtTvNOt/tF9BaSy28Owv5kioSq7RyckAYHJrh/hD8SP+E50N7bUpoBrtnxMifKZ4+MTBcYGScELkA8/KGUUAeAeBfin4g8CyiK3l+26YcBrC5diijduJjOf3bHLcgEHdkg4GPrfRNb07xHo8GraTcfaLGfd5cuxk3bWKnhgCOQRyK8P/AGgfBehabpMHiaxtPs2pXeoCK5MbEJNujZixXoGynUYzuYnJORofs1/2j/wj2ueb/wAgz7Wn2f7v+u2fvf8Aa+75PXj070AeefHnUpr74qXlvKsYSwt4beIqDkqUEuW567pGHGOAPqfc/hbqeo3HwX0m9WH7dfQ2k0cEG5YvN8p3SOPdjC8Iq7j9TnmvAPjW9vL8V9YltryC5VvKD+SSfKdY1RkY4xuBXnBOM4JyCB9J+GtNh1n4R6Rpdw0iwXuhQ28jRkBgrwBSRkEZwfQ0AfJHguCG68deHre4ijmgl1O2SSORQyuplUEEHggjjFfcdfEAhvPAfj6Fb+28y60bUEkeIEoJfLcMNpIztYAENjoQcV9twTw3VvFcW8sc0EqB45I2DK6kZBBHBBHOaAOD+Nv/ACSHXf8At3/9KI68c/Z1Mw+I10Io42Q6ZKJSzlSq+ZHyowdx3bRg44JOeMH0/wCO/iPSLPwDf6DPfRrql8kT29sAWZlWZCScDCjCtgtjODjODXGfs3eHrhtR1bxKzbbVIvsEYwD5jkpI3fI2hU7c7+vBoAx/2jv+Sh6f/wBgqP8A9Gy16n8BtNhsfhXZ3ETSF7+4muJQxGAwcxYXjptjU855J+g8s/aO/wCSh6f/ANgqP/0bLXr/AMEv+SQ6F/28f+lElAHP/tHf8k80/wD7Csf/AKKlrjP2bNNml8VazqitH5FvZC3dSTuLSOGUjjGMRNnnuOvbs/2jv+Seaf8A9hWP/wBFS1zn7M0ELXHiW4aKMzolsiSFRuVWMpYA9QCVUkd9o9KAO3+M/wAP7vxv4ftZ9L8yTVNOdjBbBkVZlkKBwSxABAUMDnsRg5BHyp/pml6j/wAt7O+tZfeOSGRT+BVgR9QRX1H49+K1z4H+I2kaPPb2jaLcW8c13Myv5sYaR0LKQSMKFDY2knBGRkEdh4r8B+HPGduU1nTo5Jwm2O7j+SePhsYcckAsTtOVzyQaAPP/AIYfG2HxJcWug+IljttUdAkN4GAju5MkYK4ARyMYHRjnG3KqfZK+FJ9N1LRvFUul27SNqllem3jazLFjMj7QY8ANncOOAelfc888Nrby3FxLHDBEheSSRgqooGSSTwABzmgD5LmZh8TfHIW/jtgbi6DRMqk3Q+1p+7XPIIOHyOcRnsTX1xXyfe+KvCHiPxTrV/b+Gru11C/cPaXJul2RspG4mJVUAum4sS0nzHjqTX1hWS/iv0X6ndU/3Gn/AI5/lTCiiitThOS/5q9/3Af/AGvXhv7R3/JQ9P8A+wVH/wCjZa9y/wCavf8AcB/9r188/HnUpr74qXlvKsYSwt4beIqDkqUEuW567pGHGOAPqcqX2vU7sd/y7/wI9P8A2b4IV8C6ncLFGJ31NkeQKNzKsUZUE9SAWYgdtx9a7D4talNpXwr8QXECxs724tyHBI2yusTHgjna5I98delZfwKsbe0+FGmzQR7JLuWeac7id7iRoweenyoo49PXNHx1vre0+FGpQzybJLuWCGAbSd7iRZCOOnyox59PXFanCeQfs831vafEqSGeTZJd6fLDANpO9wySEcdPlRjz6euK93+JXg6z8a+Emsb3U/7MjtZReC7ZQUj2KwYuCR8u1m7jHBzgYPhH7PNjb3fxKkmnj3yWmnyzQHcRscskZPHX5XYc+vriug/aJ8YXDajaeE7O522qRC4vVilB8xyfkRwBkbQofBPO9TjgGgDT8FePfh38NfB02n2niK71qdrhriQRWEkTSM21cIHAUAKoJy/Y+oFeUfE3xxD4/wDEttqsFjJZpFZR25jeQOSwLMxyAONzkD1Cg8ZwPX/gt8OPDN14MtPEeoafHqN5fpIjR3qLLFEEldfkQjAJCrknJ44wCQec/aR1mzu9c0XSIX33VhFLLcYIITzdm1Tg5DYTJBA4ZTzmgDs/2dYJofhzdPLFIiTanK8TMpAdfLjXK+o3KwyO4I7VH+0d/wAk80//ALCsf/oqWtD4A/2d/wAKvg+xf8fH2ub7d97/AF2Rjrx/q/K+7x+Oaz/2jv8Aknmn/wDYVj/9FS0AcB+zj/yUPUP+wVJ/6Nir6Xv7630zTrm/vJPLtbWJ5pn2k7UUEscDk4APSvmj9nH/AJKHqH/YKk/9GxV9F+JZLGHwrq8uqQyT6ellM11FGcM8QQ71HI5K5HUfUUAfGngT/kofhr/sK2v/AKNWvt+viDwJ/wAlD8Nf9hW1/wDRq19v0AfMH7R3/JQ9P/7BUf8A6Nlr1L4Ff8k8/wC3n/2lFXmP7RVzpF14xsVtLqSTVLa3NvewFSFiXiSIgkAEkSvnBPQdO/r3wnurO+8Paleadb/Z7GfU5ZbeHYE8uNkQqu0cDAIGBwKyn8cfmd2H/wB1rf8Abv5ne0UUVqcIUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABXJf8ACs/CH/QI/wDJmX/4uutoqZQjL4lc3o4qvQv7Gbjfeza/I5L/AIVn4Q/6BH/kzL/8XR/wrPwh/wBAj/yZl/8Ai662io9hS/lX3G/9qY7/AJ/T/wDAn/mcl/wrPwh/0CP/ACZl/wDi6P8AhWfhD/oEf+TMv/xddbRR7Cl/KvuD+1Md/wA/p/8AgT/zOS/4Vn4Q/wCgR/5My/8AxdH/AArPwh/0CP8AyZl/+LrraKPYUv5V9wf2pjv+f0//AAJ/5nJf8Kz8If8AQI/8mZf/AIuj/hWfhD/oEf8AkzL/APF11tFHsKX8q+4P7Ux3/P6f/gT/AMzkv+FZ+EP+gR/5My//ABdH/Cs/CH/QI/8AJmX/AOLrraKPYUv5V9wf2pjv+f0//An/AJnJf8Kz8If9Aj/yZl/+Lo/4Vn4Q/wCgR/5My/8AxddbRR7Cl/KvuD+1Md/z+n/4E/8AM5L/AIVn4Q/6BH/kzL/8XR/wrPwh/wBAj/yZl/8Ai662ij2FL+VfcH9qY7/n9P8A8Cf+ZyX/AArPwh/0CP8AyZl/+Lo/4Vn4Q/6BH/kzL/8AF11tFHsKX8q+4P7Ux3/P6f8A4E/8zkv+FZ+EP+gR/wCTMv8A8XR/wrPwh/0CP/JmX/4uutoo9hS/lX3B/amO/wCf0/8AwJ/5nJf8Kz8If9Aj/wAmZf8A4uj/AIVn4Q/6BH/kzL/8XXW0Uewpfyr7g/tTHf8AP6f/AIE/8zkv+FZ+EP8AoEf+TMv/AMXR/wAKz8If9Aj/AMmZf/i662ij2FL+VfcH9qY7/n9P/wACf+ZyX/Cs/CH/AECP/JmX/wCLo/4Vn4Q/6BH/AJMy/wDxddbRR7Cl/KvuD+1Md/z+n/4E/wDMKKKK1OEK5J/ANv8AbLu4t9e160+1TvcSR2t4I03sckgBfw/AV1tFTKEZbm9DE1aF/Zu1zz7UvhBoOs3C3Gqahq19OqBFkupklYLknALITjJJx7mq83wR8K3Lh55b+VwioGdomIVVCqOY+gUAAdgAK9JoqPYw7G/9pYr+b8F/kcdB8PYbW3it7fxL4khgiQJHHHfBVRQMAABcAAcYrP1L4QaDrNwtxqmoatfTqgRZLqZJWC5JwCyE4ySce5r0Gij2MOwf2liv5vwX+R5l/wAKK8H/APT3/wCQf/jdXrH4RaHpcNxDp+o6vaRXK7Z0t50jWUcjDAINw5PX1Nd/RR7GHYazPFLVT/Bf5FfT7OPTtNtbGFmaK2hSFC5yxCgAZx34qxRRWqVtDhlJybk92Fcl4f8A+SheMf8Aty/9FGutrkvD/wDyULxj/wBuX/oo1lU+KPr+jO3Cfwa/+Bf+lwOpnghureW3uIo5oJUKSRyKGV1IwQQeCCOMVX03SdN0a3a30vT7SxgZy7R2sKxKWwBkhQBnAAz7CrlFanCFZem+HNI0jVNS1LT7GO3u9TdXvHQkCVlzg7c4B+ZiSAMkknJrUooAKx9f8K6F4oihj1vTIL1Yd/lGQHKb1KtgjkZB/MKeqgjYooA87j+B3w+R5mbRZJA77lVrybEY2gbVw4OMgnnJyx5xgDqND8H6B4c0a50fS9Nji0+5dnmt5HaVZCyhWzvJyCqgY6VuUUAcv/wrjwV/Z32H/hF9K8nyvJ3fZl8zbjGfMxv3Y/izuzznPNHhv4d+FfCOoyX+h6V9kupIjCz/AGiWTKEgkYdiOqj8q6iigCOeCG6t5be4ijmglQpJHIoZXUjBBB4II4xXB/8ACkvh5/0L3/k7cf8AxyvQKKAM/RtD0vw9pyWGkWEFlarg7IUxuIAG5j1ZsAZY5Jxyaz/+EE8H/wDQqaH/AOC6H/4mugooAz9T0LR9b8r+1tKsb/yc+X9rt0l2ZxnG4HGcDp6CsPUvhj4I1W3WC48MaaiK4cG1hFu2cEctHtJHPTOOnoK6yigDL0Dw5pHhbS103RbGO0tA5fYpLFmPUszElj0GSTwAOgFWNV1Wx0PS7jU9TuY7azt03yyv0UfzJJwABySQBkmub8b/ABK8P+Avs0eqvPNdXHKWtoqvIE5+cgsAFyMcnk5wDg48Q8e/E24+KOo6Z4W0COfT7C6u0iJuZAPtLuUCeYqg7VVtxwC2eDjIGADc+AOiXeteJda8b6n5czl3iSRo0y1xIQ8jrj7hCkDgAESkA8EV9B1n6JomneHNHg0nSbf7PYwbvLi3s+3cxY8sSTySeTWhQAUUUUAFFFFAGP4p8N2fi7w5d6HfyTx2t1s3vAwDja6uMEgjqo7VX8HeDtL8D6H/AGTpPntC0rTSSTvueRzgZOAAOAo4A6euSegooAr39jb6np1zYXkfmWt1E8MybiNyMCGGRyMgnpXF/DP4Zw/Dq31FV1STUJ754y7mERKqoG2gLljnLtk59OBjnvKKACiiigDyPxz8B9F8QvJf6C8ej35QDyEiAtZCFIHyKAUJO3LLkYBO0kk1n6B+zlpGn6otxrWryataKhH2VYDbhmPQsyyE4HJwMc45xkH2yigCnpWlWOh6Xb6ZpltHbWdumyKJOij+ZJOSSeSSSck1w/i34MeGfGPiCXWr2fUre7mRVl+yzKFcqNoYhlbB2gDjA4HGck+iUUAeRz/s6+DZriWVLvWYEdyyxR3EZVAT90boycDpySfUmvQPB/hqHwf4TsNBguJLhLRGBmcAF2Zi7HA6DcxwOcDHJ61uUUAeZ+O/gzpvjfVLnVpda1K31CRESLcVlghVcDCx4BwfmOAw+ZifauX1L9mnTZbhW0vxJd20GwBkurZZ2LZPIZSgAxjjHY888e6UUAV7+xt9T065sLyPzLW6ieGZNxG5GBDDI5GQT0rw/wD4Zos/7R3/APCTz/YfNz5P2MeZ5efu79+N2ON23GedvaveKKAPC9S/Zp02W4VtL8SXdtBsAZLq2Wdi2TyGUoAMY4x2PPPFz/hm7w3/AGjv/tnVfsPlY8nMfmeZn72/bjbjjbtznnd2r2iigDi5Phlosvw8h8FNdakNNifesqzhZSfMMnzELtYZY8FSOh6gEcn/AMM4+D/+glrn/f8Ah/8AjVewUUAcHafCzTdK+H2reEdL1TUoINRdpGuXdWkRiEGPlCgoQgBXuCwyM8cvP+zf4Va3lW31XWY5yhEbySROqtjglQgJGe2Rn1FeyUUAcX4H+GWi+ALi8n0q61KZ7tFRxdTgqApJGFVVBPPUgkc4xk58M+N11/wlXxVGnaJbz311Z2i2bx2yeaXkUvIwULknaGwe4KtxxX0P451PUdF8Da1qWkw+bfW1o8kfzKvl4HMnzAg7Bl9pHzbcd68Q/Zr/ALO/4SHXPN/5Cf2RPs/3v9Tv/e/7P3vJ68+negDyufwX4qtbeW4uPDWswwRIXkkksJVVFAySSVwABzmtC81b4g+Gre1t73UPE+lQbNltHNNcQLtUAYQEgYAIGB0yK+06jnghureW3uIo5oJUKSRyKGV1IwQQeCCOMUAfLem/F7X/ABBpN/4e8QzR3gu0DQXCwKjq6ujbW27V2bUf+EnJHOOn1RXxGn9nf8J7df2R/wAgz7XP9j+9/qfm2fe+b7uOvPrX25WS/iv0X6ndU/3Gn/jn+VMKKKK1OE828O+MNB8P3niO01S++zzvrVzIq+S75UkDOVUjqDWhq3jvwVrOjX2l3GsyLBe28lvI0dvKGCupUkZQjOD6Gu5orGMKkVZNfd/wT062KwlaftJ05Xfaat/6Qz5z+EeneGPB2o3Gt67rUEmpDfBaxRW0rpGmcGXcUzuYDjGMKxB5YhfX/wDhZnhD/oL/APktL/8AEV1tFO1Xuvu/4Jl7TA/8+5/+Br/5WfPfxd+weP8AUdMm0zxVYx2tpE6G3u7adNrsQS4ZYmLbgFGDjGwYzuNeb/8ACvf+pp0P/vm7/wDjFfZlFFqvdfd/wQ9pgf8An3P/AMDX/wArPB/BMmlaD8PtY8M6t42jnTUbeWGGKKxneOyLh1YozKC4O5WK4UAg46knyeTwxqnh54dQ0fxDaTXiPtU6dJPFLGCpBbc8aADHBwc/N0xmvtCii1Xuvu/4Ie0wP/Puf/ga/wDlZ8Xz6d4h8X6zLd+INYjjnKEi5v5HdfvZ2KI1cqMsxAACjnpwK9z1zxP4c0r4Z3ugeBdW+wXUcTCzUJPldz73Cu6khmBcAkjBYcrjI9dootV7r7v+CHtMD/z7n/4Gv/lZ8Xw+AXlQs/iLRoSHZdri5JIDEBvlhIwQMjvgjIByB3nwttE8DeKGvrzxjYjTZImW4tbSG5k+0HHyAholC7SS24ZPBHRjX0pRRar3X3f8EPaYH/n3P/wNf/Kzwf4rWvg3x8kGoafr8dtrVughVp4J/Kli3E7WwhKkFmIIHcgg5BXzfSNV8b+BXu7Dw5rMbWkrq7PAqvE7beqrMgKnsTtGdo6gCvsCii1Xuvu/4Ie0wP8Az7n/AOBr/wCVnxfpvh698S+IGl8T63JZJIhabULrfdyMQAFUBSSx6DkgAA88AH6LPjPwhpPhKbSPDGqQafJFaPFYZtZSkUhU7WbKHPzHJJBJ5JyTXotFFqvdfd/wQ9pgf+fc/wDwNf8Ays+O77wZeaneSXl/4z0q7upMb5p2vJHbAAGWMGTgAD8K7D4W2ieBvFDX154xsRpskTLcWtpDcyfaDj5AQ0ShdpJbcMngjoxr6UootV7r7v8Agh7TA/8APuf/AIGv/lZ8r/EHTrvxZ4smvB4ztL3T5LhjapcJcILOMrnGwRkAfKqZXJY7WI5JB4D0J/CXjGw1Y+NNNt7eN8XItYbl2mi6tGVaFQQ2AMk8cMOVFfVFFFqvdfd/wQ9pgf8An3P/AMDX/wArPmP41/ZfF3iGz1nQ7+C7jjtEtWg2yRyAh5GLfOoXb8wH3s5PTHNcfpOr+OdB0OfRtJvZ7Kxnl86RYHjR9/yjIkHzrwijgj9TX2ZRRar3X3f8EPaYH/n3P/wNf/Kz5L8C+CPDEsovfGes+RCMFNPtklLsQ3IkcIQFIHRDk7uqkYPu+u+OfCGt+HtT0n+3fJ+3Wktt5v2SVtm9Cu7G0ZxnOMiu9ootV7r7v+CHtMD/AM+5/wDga/8AlZ8S6doV7p+uqziOSCJ3Xz42+VxggMAcNg+4B55Ar7aoopwhJScpP+tfN9ycRiKU6UaVKLSTb1ae6iukY/yhRRRWhxnk3xJ8Q3vhrxNe3ultGupvoSw2m9Nw3tc8nnjIQOwzxlRnPQ/PGq2niLXNUuNT1PzLm8uH3yyvImWP54AAwABwAABgCvuCisVCab5WtfL/AIJ6MsThakYqrTk2klpJJaeTg/zPlv4Y+NfFHg3UbLTbz994caXbNDK6n7OHI3SIRlhjltoyDluMncM/4k+IvFHjTXLuF5vN0S3u5DYQR7Y02D5VcgncWKjPzdNzYCg4r60op2q9193/AASPaYH/AJ9z/wDA1/8AKz4n8ODxF4W8QWetabbxi7tXLJ5hRlYEFWUjPQqSOMHngg816B8VbLR/FlvZeINP1S0fxEtvFFqFrBbyRxXDAcvGzKDkE4+c8oF5BXDfTFFFqvdfd/wQ9pgf+fc//A1/8rPifTZPGWjW7W+l6jqVjAzl2jtb4xKWwBkhWAzgAZ9hVzU/AcFno8VzZeI7HUL7nzrOOCZMfMAux3QBuCSd2zGMDdX2ZRRar3X3f8EPaYH/AJ9z/wDA1/8AKz438BaEbbxrpd/rF3Hp1nY3Ed20rKZS5jYMEVUyckjqcADJ5OAdT4k+IvFHjTXLuF5vN0S3u5DYQR7Y02D5VcgncWKjPzdNzYCg4r60ootV7r7v+CHtMD/z7n/4Gv8A5WfE+gN4r8LaoupaLJJaXYQpvV42DKeoZWyGHQ4IPIB6gV6h8VvH2va6kGleHbqNdLuLIC/FvkF5WY7498iqxQADkKu4MwOc4H0RRRar3X3f8EPaYH/n3P8A8DX/AMrPheDRtatbiK4t45IZ4nDxyRzKrIwOQQQcgg85r6b8F/FK1m8L2p8W3sFvq65SQQxSNvAOA7BU2qx6kKSO/Gdq+m0UWq9193/BD2mB/wCfc/8AwNf/ACs+L/GWkXN9411u9sDHdWl1ey3EUyHaCrsXAw+CCN2Dx1BxkYJ+hvgdG8PgF4nGHS6KsPQiKOvSqKFCfMnJ7eX/AARyxOHVGVOjBrmtq5J7ekV+YUUUVqcAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFfLvjazk0n4aeG9a060jje7uJ47y8Llm3Bm8tApbGCquSQv8A5GefNf+Eg1T/n6/8AIa/4VhGdSUVJRWvn/wAA9OvhcHQqypSqSvFtfAujt/OfdlFfDdr4pv7fz/NSC68yJo185SPKY9HXYV+Ydt2V55Br1f4CavcS+J3+0Rz3DzxSwCSKMbYgAj7nxjC/Ltz/AHmUd6bnNNcyX3/8AmOGwtSMnSqSbSb1iktPNTf5H0bRRRWx5wUVyXxM/wCSe6p/2y/9GpR/wrPwh/0CP/JmX/4uspTlzcsV+Pr5Psd1PD0PYKtWm1dtJKKeyi+so/zHW0VyX/Cs/CH/AECP/JmX/wCLo/4Vn4Q/6BH/AJMy/wDxdF6vZff/AMAPZ4H/AJ+T/wDAF/8ALDraK5L/AIVn4Q/6BH/kzL/8XR/wrPwh/wBAj/yZl/8Ai6L1ey+//gB7PA/8/J/+AL/5YdbRXJf8Kz8If9Aj/wAmZf8A4uj/AIVn4Q/6BH/kzL/8XRer2X3/APAD2eB/5+T/APAF/wDLDraK5L/hWfhD/oEf+TMv/wAXR/wrPwh/0CP/ACZl/wDi6L1ey+//AIAezwP/AD8n/wCAL/5YdbRXJf8ACs/CH/QI/wDJmX/4uj/hWfhD/oEf+TMv/wAXRer2X3/8APZ4H/n5P/wBf/LDraK5L/hWfhD/AKBH/kzL/wDF0f8ACs/CH/QI/wDJmX/4ui9Xsvv/AOAHs8D/AM/J/wDgC/8Alh1tFcl/wrPwh/0CP/JmX/4uj/hWfhD/AKBH/kzL/wDF0Xq9l9//AAA9ngf+fk//AABf/LDraK5L/hWfhD/oEf8AkzL/APF14x8W/C+v+H9Y8zQNC8vQliaUXVmks7ABV3+eWLBNpDEEAAqepIOC9Xsvv/4AezwP/Pyf/gC/+WH0pRXxX4T1O81PxlodheTeZa3WoW8MybQNyNIoYZAyMgnpX1P/AMKz8If9Aj/yZl/+LovV7L7/APgB7PA/8/J/+AL/AOWHW0VyX/Cs/CH/AECP/JmX/wCLrO0bQtN8P/FJ7TS7b7PA+imRl3s+WM4GcsSegFJzmmuZL7/+AXHDYWpGTpVJNpN6xSWnmpv8jvqKKK2POCiivNvB/h248QeFrLVLvxP4jSeffuWK/IUbXZRjIJ6Ad6znNpqKVzrw+GhUpyq1J8qTS2vq7v8AQ9Jorkv+EE/6mvxR/wCDH/7Go5/BLQ28sqeJPFc7ohZYo9RUM5A+6NwAyenJA9SKXPP+X8S/YYX/AJ/f+Ss7Givj/VfiP4207VLi1nu9WsHR8i2u7mYSxqeVDZK5O0jnaM9cDNeifB+91j4gf2z/AGt4i1qH7D5Hl/ZL11zv8zOd27+4OmO9HPP+X8Q9hhf+f3/krPfKK5L/AIQT/qa/FH/gx/8AsaP+EE/6mvxR/wCDH/7Gjnn/AC/iHsML/wA/v/JWdbRXJf8ACCf9TX4o/wDBj/8AY0f8IJ/1Nfij/wAGP/2NHPP+X8Q9hhf+f3/krOtorkv+EE/6mvxR/wCDH/7Gj/hBP+pr8Uf+DH/7Gjnn/L+Iewwv/P7/AMlZ1tFcl4Be4+x61b3F7dXf2XVp7eOS6lMj7FCgAk/n+JrrauEuaNzDE0PYVXTvewUUVyXiu71j/hIdA0rStU/s/wC3/aPMl+zpL9xFYcN+I6jrROXKrhhqDr1ORNLRu7vayTb2Tey7HW155F4m0fw58QvFX9q3n2fz/snl/u3fdti5+6DjqK1v+Ef8X/8AQ8f+UmL/ABo/4R/xf/0PH/lJi/xrGbnKzUXp6f5npYWnhqXPGpVi1JW050909/Zvt2D/AIWZ4Q/6C/8A5LS//EUf8LM8If8AQX/8lpf/AIij/hH/ABf/ANDx/wCUmL/Gj/hH/F//AEPH/lJi/wAanmr9vwX/AMkX7DK/5/8AyaX/AMoD/hZnhD/oL/8AktL/APEUf8LM8If9Bf8A8lpf/iK8Ivvjv4jjvJFsNQnntRjZJPawRO3AzlQrAc5/iPrx0rrPht4/8U/EHWbvS216TT54bf7QjCyhmV1DBWB4Ug5ZccHPPTHJzV+34L/5IPYZX/P/AOTS/wDlB6X/AMLM8If9Bf8A8lpf/iKP+FmeEP8AoL/+S0v/AMRR/wAI/wCL/wDoeP8Aykxf40f8I/4v/wCh4/8AKTF/jRzV+34L/wCSD2GV/wA//k0v/lAf8LM8If8AQX/8lpf/AIij/hZnhD/oL/8AktL/APEUf8I/4v8A+h4/8pMX+NH/AAj/AIv/AOh4/wDKTF/jRzV+34L/AOSD2GV/z/8Ak0v/AJQH/CzPCH/QX/8AJaX/AOIo/wCFmeEP+gv/AOS0v/xFcv491fxH4B8PDVrzxbPdeZL9nhih0qAZkKMy7iW+VfkIJAJGehryj/hfXi//AJ+f/IcX/wAbo5q/b8F/8kHsMr/n/wDJpf8Ayg9//wCFmeEP+gv/AOS0v/xFH/CzPCH/AEF//JaX/wCIrn/BM/i/xj4Qsdf/AOEs+x/avM/cf2dFJt2yMn3uM5256d66D/hH/F//AEPH/lJi/wAaOav2/Bf/ACQewyv+f/yaX/ygP+FmeEP+gv8A+S0v/wARR/wszwh/0F//ACWl/wDiKP8AhH/F/wD0PH/lJi/xo/4R/wAX/wDQ8f8AlJi/xo5q/b8F/wDJB7DK/wCf/wAml/8AKA/4WZ4Q/wCgv/5LS/8AxFH/AAszwh/0F/8AyWl/+Io/4R/xf/0PH/lJi/xo/wCEf8X/APQ8f+UmL/Gjmr9vwX/yQewyv+f/AMml/wDKDyD4tWOkeO/ENlq2k+JrGLyrQW0kV3b3CY2uzBgVjbOd5GCBjA6541PhnpngHwK51O+16PUNadAqyizmCWwK/MseVySTkbzgkcALls+l/wDCP+L/APoeP/KTF/jR/wAI/wCL/wDoeP8Aykxf40c1ft+C/wDkg9hlf8//AJNL/wCUB/wszwh/0F//ACWl/wDiKP8AhZnhD/oL/wDktL/8RR/wj/i//oeP/KTF/jR/wj/i/wD6Hj/ykxf40c1ft+C/+SD2GV/z/wDk0v8A5QH/AAszwh/0F/8AyWl/+Io/4WZ4Q/6C/wD5LS//ABFH/CP+L/8AoeP/ACkxf40f8I/4v/6Hj/ykxf40c1ft+C/+SD2GV/z/APk0v/lAf8LM8If9Bf8A8lpf/iKP+FmeEP8AoL/+S0v/AMRR/wAI/wCL/wDoeP8Aykxf41b8Daneaz4NsL+/m866l8ze+0LnEjAcAAdAKcZ1XLleny/+2Jq4fAQpOrC8kmk7T7ptb0l2ZU/4WZ4Q/wCgv/5LS/8AxFH/AAszwh/0F/8AyWl/+IrraK0tV7r7v+CcftMD/wA+5/8Aga/+VnJf8LM8If8AQX/8lpf/AIij/hZnhD/oL/8AktL/APEV0Wq339maRe3/AJfmfZYJJtm7G7apOM9s4rmLXxT4pvbOC7t/BW+CeNZI2/tWIblYZBwRnoaiU5xdm/8AyVv9Tqo4fDVoOcabttrVhHX5xRL/AMLM8If9Bf8A8lpf/iKP+FmeEP8AoL/+S0v/AMRR/wAJB4v/AOhH/wDKtF/hR/wkHi//AKEf/wAq0X+FT7Sff/yWX+Zp9Sw/8n/lel/8iH/CzPCH/QX/APJaX/4ij/hZnhD/AKC//ktL/wDEUf8ACQeL/wDoR/8AyrRf4Uf8JB4v/wChH/8AKtF/hR7Sff8A8ll/mH1LD/yf+V6X/wAiH/CzPCH/AEF//JaX/wCIo/4WZ4Q/6C//AJLS/wDxFH/CQeL/APoR/wDyrRf4Vyev/GseF9UbTNY0GOC8VA7RLqAlKA9N2xCAcc4POCD0Io9pPv8A+Sy/zD6lh/5P/K9L/wCROs/4WZ4Q/wCgv/5LS/8AxFH/AAszwh/0F/8AyWl/+Irk9A+NY8UaoumaPoMc94yF1ibUBEXA67d6AE45wOcAnoDXWf8ACQeL/wDoR/8AyrRf4Ue0n3/8ll/mH1LD/wAn/lel/wDIh/wszwh/0F//ACWl/wDiKP8AhZnhD/oL/wDktL/8RR/wkHi//oR//KtF/hR/wkHi/wD6Ef8A8q0X+FHtJ9//ACWX+YfUsP8Ayf8Alel/8iH/AAszwh/0F/8AyWl/+Io/4WZ4Q/6C/wD5LS//ABFH/CQeL/8AoR//ACrRf4Uf8JB4v/6Ef/yrRf4Ue0n3/wDJZf5h9Sw/8n/lel/8iH/CzPCH/QX/APJaX/4ij/hZnhD/AKC//ktL/wDEVhz/ABTmtbiW3uNN0aGeJykkcniS1VkYHBBBOQQeMVc0bx5rfiHTkv8ASPC0F7atgb4dYhO0kA7WGMq2CMqcEZ5FHtJ9/wDyWX+YfUsP/J/5Xpf/ACJof8LM8If9Bf8A8lpf/iKP+FmeEP8AoL/+S0v/AMRR/wAJB4v/AOhH/wDKtF/hR/wkHi//AKEf/wAq0X+FHtJ9/wDyWX+YfUsP/J/5Xpf/ACIf8LM8If8AQX/8lpf/AIij/hZnhD/oL/8AktL/APEUf8JB4v8A+hH/APKtF/hWj4X1+bxBZ3klxYfYZ7S7e0kh84S4ZQufmAA6nH4U1Obdr/8Akr/zJqYXD04Oo6baXarB/lFmVP8AEXwVdW8tvcalHNBKhSSOS0lZXUjBBBTBBHGK+cPE3h6PQPEo1DwPrck0EjyPEbfzbaWzBJAj3OcsNrY3A5PzZA4z9gUVpar3X3f8E4/aYH/n3P8A8DX/AMrPl/Tfiz8ULG4aW4ktNQQoVEV1BEqg5HzDyyhzxjrjk8dMR+J/ib8QPFGhzaRNaWNlbz8TNZZR5E7oS0jfKe+MZ6ZwSD9M6nqdno2nS39/N5NrFje+0tjJAHABPUiud/4WZ4Q/6C//AJLS/wDxFRKcou0ppfL/AIJ00cPSrx5qWHqSXlJP8qZ41o+j+FvC3gXWIINcj1PXtRSBSVs3jWJVdWZEZlyRkEkkjdtX5QRz9JVyX/CzPCH/AEF//JaX/wCIo/4WZ4Q/6C//AJLS/wDxFKFSCk5Smv6v5vuXiMHiZ0o0qWGmkm3qm91FdIx/lOtorkv+FmeEP+gv/wCS0v8A8RR/wszwh/0F/wDyWl/+IrT29L+Zfecf9l47/nzP/wABf+R1tFcl/wALM8If9Bf/AMlpf/iKP+FmeEP+gv8A+S0v/wARR7el/MvvD+y8d/z5n/4C/wDI62iuS/4WZ4Q/6C//AJLS/wDxFH/CzPCH/QX/APJaX/4ij29L+ZfeH9l47/nzP/wF/wCR1tFcl/wszwh/0F//ACWl/wDiKP8AhZnhD/oL/wDktL/8RR7el/MvvD+y8d/z5n/4C/8AI62iuS/4WZ4Q/wCgv/5LS/8AxFH/AAszwh/0F/8AyWl/+Io9vS/mX3h/ZeO/58z/APAX/kdbRXJf8LM8If8AQX/8lpf/AIij/hZnhD/oL/8AktL/APEUe3pfzL7w/svHf8+Z/wDgL/yOtoqG1uob2zgu7d98E8ayRtgjcrDIODz0NTVqcTTTswooooEFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRWdda/o1lcPb3er2EE6Y3Ry3KIy5GRkE56EGov+Ep8Pf9B7S/8AwMj/AMannj3NlhqzV1B/czWorJ/4Snw9/wBB7S//AAMj/wAaP+Ep8Pf9B7S//AyP/Gjnj3H9Vr/yP7ma1FZP/CU+Hv8AoPaX/wCBkf8AjR/wlPh7/oPaX/4GR/40c8e4fVa/8j+5mtRWT/wlPh7/AKD2l/8AgZH/AI0f8JT4e/6D2l/+Bkf+NHPHuH1Wv/I/uZrUVk/8JT4e/wCg9pf/AIGR/wCNH/CU+Hv+g9pf/gZH/jRzx7h9Vr/yP7ma1FZP/CU+Hv8AoPaX/wCBkf8AjR/wlPh7/oPaX/4GR/40c8e4fVa/8j+5mtRWT/wlPh7/AKD2l/8AgZH/AI0f8JT4e/6D2l/+Bkf+NHPHuH1Wv/I/uZrUVk/8JT4e/wCg9pf/AIGR/wCNH/CU+Hv+g9pf/gZH/jRzx7h9Vr/yP7ma1FV7PULLUYTNY3cF1ErbS8EgdQeuMg9eR+dWKpO+xjKLi7SVmFFFFAgooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigDkvhn/AMk90v8A7a/+jXrxH9oLwtpug6zot7pdraWUF3byRNbWtssSho2B3nbgEkSgdP4Bye3t3wz/AOSe6X/21/8ARr14b+0d/wAlD0//ALBUf/o2WsqH8KPojuzT/fq3+OX5s5P4TaVY618T9FsNSto7q0d5HeGTlWKRO65HcblHB4PQ5HFfSWm6TpujfFVrfS9PtLGBtELtHawrEpbzwMkKAM4AGfYV88/BL/kr2hf9vH/pPJX0l/zV7/uA/wDteir09QwP/Lz/AAP9DraKKK1OE5L4mf8AJPdU/wC2X/o1K62uS+Jn/JPdU/7Zf+jUr5t/4Xb8Q/8AoYf/ACSt/wD43WS/iv0X6ndU/wBxp/45/lTPr+ivL/gz488QeOdO1KTW7eAR2XkxRXUMLJ57kNv3HO0sMISFAxu6YIrn/iR8YvFnhDxRLYWnh2C3sF+SG41CF2+0kAFnRkcLt+ZeOSO+Cdo1OE9worwfwL8avFni3xRpujjw/YzrJKxvJLbenlwYUb/mYhdpyTnO7KqADyfRPij4v1LwR4OOr6Xp8d3P9oSJjKrGOFWz87hcHGQFHI5cc9iAdpRXh/w3+NXiDxb4oi0a+8PwXCzcmbT9yfZkAO53DsQVyVHVcdgxIWvUPGPjHS/A+h/2tq3ntC0qwxxwJueRzk4GSAOAx5I6euAQDoKK+cNZ/aR1h9Rf+w9GsYrEZC/bg8kj8nDHYyhcjHy/Ng5+Y13fhL48eGfEVxFZ6ikmi3bozF7qVTbgg8KJcjkrz8yqM5GScZAPVK4f4q+N7zwF4Sj1OwtYLi6mu0tkE5OxMqzFiAQTwhGMjrntg9xXzZ8ePiHY686eFdNSRhp167Xk0ke0GVFKBUOckAtIGyo5AwSOSAd38Jfi1eePNRvdJ1awggvoYjcxy2gIjaMFVKkMxIYFgcg4IJ6Y+b1ivkz4QfEnTfh9casuqWd3PBfJEVe12syMhbgqxAIIc854wODnj6X8KeLdI8Z6MNU0aaSSAP5ciyRlGjk2qxQ54JAYcgkehNAG5RXJ+M/iN4c8Coi6vcyNdypvis7dN8rruAzjICjryxGdrYyRivK/+Gmv+pR/8qX/ANqoA+gK5/x3/wAk88S/9gq6/wDRTVh+FPi/4Q8W3AtLe8ksbxn2x21+qxNLyoG0glSSWwFzuODxjmtzx3/yTzxL/wBgq6/9FNQB8geBP+Sh+Gv+wra/+jVr7fr4w+Fms2egfEzRNRv38u1WVonkJACeYjRhmJIAUFwSewBr2/xJ+0N4a0zy00O1n1qRsFm+a2jUc5GXXcWGBxtxg9cjFAHsFcl/zV7/ALgP/tesjwT8ZPDnjS/g0uOO7sdUlQFbeaPcsjBGZwjrkYUKeWC54wOw1/8Amr3/AHAf/a9ZVenqd2B/5ef4H+h1tFFFanCFcF4Q1P8AsT4ODVvJ877DaXVz5W7bv2PI23ODjOMZwa72vn/UfiZovh/4SDwyBJd6tfWVzCYoiNtuJHdQZG7HaxYKAScDO0MDWT/ir0f6HdT/ANxqf44flUDwP8cPFnifxppeiy6RpUsN3LskEAeJ1QAlnDM5HygFsY524HJFfQFfEHgfVtL0HxpperazbT3FjZy+c0cH396gmMj5l6PtPJ7d+lfW/h74i+EfFV4bPR9bgnuh0hdXid+CflVwC2ApJ25x3xWpwnk/7TX/ADK3/b3/AO0ak/Zmnma38S27SyGBHtnSMsdqswlDEDoCQqgnvtHpW5+0d/yTzT/+wrH/AOipay/gVqmkeGvhbrWvao8dtBHqeye5ERZtuyIIDtBYgNIcDtuPqaAPdKK8bP7SHhX7QirpWsmAoxdzHEGDZG0Bd+CCN2TkYwODnj1DQPEekeKdLXUtFvo7u0LlN6gqVYdQysAVPQ4IHBB6EUAalFcP4h+LXhHwx4jGh6neTpdL/wAfDpbuyW/yB13EDJ3BhjYG98Vj6N8fPBWr6ilnK19pu/AWa+iVYyxIABZGbb1zlsKADkigD1Ciq9/fW+madc395J5draxPNM+0naigljgcnAB6V5nqX7QXgixuFitzqWoIUDGW1tgqg5PynzGQ54z0xyOeuADp/An/ADMv/Yeuv/Za62uK+G19b6np2uX9nJ5lrdazPNC+0jcjBCpweRkEda7WsqPwI7sy/wB6l8vyQVyXiD/koXg7/t9/9FCutrkvEH/JQvB3/b7/AOihRW+H5r80GXfxn/gqf+kSOtorD8UeLtF8HaW9/rF7HCAjNFAGBlnIwNsaZyxyy+wzkkDmvO/+GjvB/wD0Ddc/78Q//Ha1OE9gorzPTfjz4Dvrdpbi+u9PcOVEV1aOzEYHzDy94xzjrng8dM1/BHxph8beMTodv4eu4IGSR47oSiTaq9DIoUBARxnc2GKjnOaAOb+OHw60LTPCT6/oeiQWl1HdxfangYxosJUpxHnaMuY/urnnPqawP2bvsf8AwmWrb/P+3f2f+6248vy/MTfu77s+XjHGN2e1ej/H7+zv+FXz/bf+Pj7XD9h+9/rsnPTj/V+b97j8cVwn7NOmwy6zr+qM0nn29vFbooI2lZGZmJ4znMS457nr2APe9duryx8Paneadb/aL6C0llt4dhfzJFQlV2jk5IAwOTXy54D8e+N9R+INhGPE0jve3GJIdRuitvIM7zGAVYRltuxSi5BYAdcV9R6nruj6J5X9rarY2HnZ8v7XcJFvxjONxGcZHT1Fcv4Osvh1Z+IdR/4RF9KbU5ohNcfY5/NxGXPCclVXcBlUwB8mQPloA7iiiqepatpujW63GqahaWMDOEWS6mWJS2CcAsQM4BOPY0ASX1hZ6nZyWd/aQXdrJjfDPGJEbBBGVPBwQD+FfMHx70Pw/oPi2xi0aw+xXE9p51zFCipBjcVQoo+63yNuwAPunqWNe96b8TvBGq27T2/ifTURXKEXUwt2zgHhZNpI564x19DXin7SP2P/AITLSdnn/bv7P/e7seX5fmPs2992fMznjG3HegD1f4Jf8kh0L/t4/wDSiSsv42fEDV/BOl6Zb6L5cN3qLyH7UyhzEsezIVWBBLbwMnOADxkgjY+DME1r8KdHt7iKSGeJ7lJI5FKsjC4kBBB5BB4xXQeLdL8Oat4flh8UpaHS0dXd7qXyljbOFIfIKnJxkEZ3Ed8UAeV/ATx14g1+W+0PVjPf29rF58eoTMzvGSwHlOxzuzliuTkbWHIxt9wrm/Bnh7wpoWlvL4SgtBZ3b72nt5zOJSuV/wBYWYkAgjGcA7u5NdJQAUVHBPDdW8VxbyxzQSoHjkjYMrqRkEEcEEc5rHn8aeFbW4lt7jxLo0M8TlJI5L+JWRgcEEFsgg8YoA3KKKKACiqepatpujW63GqahaWMDOEWS6mWJS2CcAsQM4BOPY1zdv8AFTwNc/Y/L8TWI+2b/L8xjHt2dfM3AeX7b9u7tmgDsKK4/wAQ/E/wn4d0MatJqsF/C8vkxR6fKk7yOMFgMNj5QwJyRjI7kA1/h/8AFDR/iB58FnBPaX1tEkk0E7JzuyG8sg5ZVOAWIH3l4GcUAdxXJfDP/knul/8AbX/0a9dbXJfDP/knul/9tf8A0a9ZP+KvR/od1P8A3Gp/jh+VQ62iiitThMnxT/yKGtf9eE//AKLajwt/yKGi/wDXhB/6LWjxT/yKGtf9eE//AKLasfQ/FnhvTPDmkWd/4g0q0uo7C33wz3scbrmJSMqTkZBB/Gsv+XvyO7/mB/7f/Q66iiitThCiiigAr58+NXwpsdPsNR8aaVcSRO1x519aytuVjI6rujOMg72yQSR8xwRgA/Qdef8Axt/5JDrv/bv/AOlEdAHiHwB0z7f8UILnzvL/ALPtJrnbtz5mQItuc8f63Oefu475H1fXy5+zrBDN8Rrp5Yo3eHTJXiZlBKN5ka5X0O1mGR2JHevqOgAooooA8n8efHGz8GeI5tDh0Oe/urfb9od5xCg3IrrtIDFuG5yFxjvXokE8PifwrFcW8t3aQapZB45I2Ec8SypkEEZCuA2c84I715v47+BsPjPxZc69Fr8li9yiCWFrUTDcqhMqd64G1V4Oec884HqGlaVY6HpdvpmmW0dtZ26bIok6KP5kk5JJ5JJJyTQB8cfEPwPN4A8SrpMt9HepJbpcRTLGUJUllwy5ODuVuhPGD3wPc/2cf+Seah/2FZP/AEVFXMftI+HrhdR0nxKrbrV4vsEgwB5bgvIvfJ3Bn7cbOvIr0f4Jf8kh0L/t4/8ASiSgD0CiiigArkvAn/My/wDYeuv/AGWutrkvAn/My/8AYeuv/Zayn8cfmd2H/wB1rf8Abv5nW0UUVqcJyXxM/wCSe6p/2y/9GpXW1yXxM/5J7qn/AGy/9GpXW1kv4r9F+p3VP9xp/wCOf5UwooorU4QooooAKy/Eev2Phbw/ea1qTSC0tUDP5a7mYkhVUD1LEDnA55IHNalcv8QvCf8Awmvgu+0aN4I7p9sltNMm4RyKQR7rkZUsOQGPB6EA5vwR8atF8aeIDoq2N3Y3cryG08zDrMijcMkfccqGO3kDb94kgV6ZXg/w9+But+GvGljrWr6lYm3s90ipZTS73cggAnamF5JPJzjaQQxr2DxZfXGmeDdcv7OTy7q10+4mhfaDtdY2KnB4OCB1oA2Kjnjaa3liSaSB3QqssYUshI+8NwIyOvII9Qa+TPhp4h8Wa18UPD8X/CQ300nmsH+2XTyqYMb5kw277yx+n3gpyCAR9b0AfEnjPwNrXgXVEsdYijIlTfDcQEtFMOM7SQDkE4IIBHB6EE+t/sy/8zT/ANun/taqn7S2mwxazoGqK0nn3FvLbupI2hY2VlI4znMrZ57Dp31/2adNmi0bX9UZo/IuLiK3RQTuDRqzMTxjGJVxz2PTuAe6Vk+Kf+RQ1r/rwn/9FtWtWT4p/wCRQ1r/AK8J/wD0W1TP4Wb4X+PD1X5h4W/5FDRf+vCD/wBFrWtWT4W/5FDRf+vCD/0Wta1EPhQYr+PP1f5hRRXD+M9Ms9Z8ZeE7C/h861l+2b03Fc4jUjkEHqBSqScY3Xl+LKwlCNerySdlaTva/wAKb2uu3c7iiuS/4Vn4Q/6BH/kzL/8AF0f8Kz8If9Aj/wAmZf8A4upvV7L7/wDgGvs8D/z8n/4Av/lh1tFcl/wrPwh/0CP/ACZl/wDi6P8AhWfhD/oEf+TMv/xdF6vZff8A8APZ4H/n5P8A8AX/AMsOtorkv+FZ+EP+gR/5My//ABdH/Cs/CH/QI/8AJmX/AOLovV7L7/8AgB7PA/8APyf/AIAv/lh1tFcl/wAKz8If9Aj/AMmZf/i6P+FZ+EP+gR/5My//ABdF6vZff/wA9ngf+fk//AF/8sOtorkv+FZ+EP8AoEf+TMv/AMXR/wAKz8If9Aj/AMmZf/i6L1ey+/8A4AezwP8Az8n/AOAL/wCWHW0V8h/FKV/DnxH1bSdJb7PYweT5cWN+3dCjHlsk8knk17P8OvBeia/8P9H1TWtF2X9xETIfMmj3gMwV9u7+JQrccHdkADFF6vZff/wA9ngf+fk//AF/8sPVaK5L/hWfhD/oEf8AkzL/APF0f8Kz8If9Aj/yZl/+LovV7L7/APgB7PA/8/J/+AL/AOWHW0VyX/Cs/CH/AECP/JmX/wCLo/4Vn4Q/6BH/AJMy/wDxdF6vZff/AMAPZ4H/AJ+T/wDAF/8ALDraK5L/AIVn4Q/6BH/kzL/8XR/wrPwh/wBAj/yZl/8Ai6L1ey+//gB7PA/8/J/+AL/5YdbRXJf8Kz8If9Aj/wAmZf8A4uj/AIVn4Q/6BH/kzL/8XRer2X3/APAD2eB/5+T/APAF/wDLDraK5L/hWfhD/oEf+TMv/wAXR/wrPwh/0CP/ACZl/wDi6L1ey+//AIAezwP/AD8n/wCAL/5YdbRXm3iLwfoPh+88O3el2P2ed9ato2bznfKkk4wzEdQK9Jpwm22pLYjE4enThCpSk2pX3Vnp6NhRRRWhyHzf8XtK8San4y1X+wrC7u4Y5bf7R9jh8yRWMA2fdG8KQHzj5chc87a8Z/tK+/5/bj/v63+NfZPh/wD5KF4x/wC3L/0Ua+QfEumw6N4q1fS7dpGgsr2a3jaQgsVRyoJwAM4HoKwpRi4ttdX+bPTx1erCrGMZNLkh1f8AJE+sfh/oemaj8PtAu9S8Oael3JZRlzJDHK0gxhZC2OS6gPg8jdg8isvxv4r+HvgLUbaw1Pw5DPdTxecEtNOhbYmSASWKjkhumfunOOM9d4E/5J54a/7BVr/6KWvnT40+G9am+Ld28Gl3c41NI3svIiMhnCQor7QuTlSpyOoGD0IJ15I9jj+tV/5397PefCkngfxnow1TRtH0+SAP5ciyaeiNHJtVihyuCQGHIJHoTW5/wi3h7/oA6X/4Bx/4V5j+z/4R1rw5pesX+sWUlkNQeJYYJ1KS4j8wFmQjKgl+M8nBOMYJ9ko5I9g+tV/5397Mn/hFvD3/AEAdL/8AAOP/AAo/4Rbw9/0AdL/8A4/8K1qKOSPYPrVf+d/ezJ/4Rbw9/wBAHS//AADj/wAK+O/GN/rMfjHVVvbaXSJxcN/oEZCLbr/Co2AKRtx8wHzfe5zmvtqvH/2idE+3eBrTVo7ffNpt2N8u/HlwyDa3GecuIh0JH0zRyR7B9ar/AM7+9nz74a1DUW8VaQqLLqDm9hC2Uk+1bg7x+7JbgBumTxzzX2Z/wi3h7/oA6X/4Bx/4V8beBP8Akofhr/sK2v8A6NWvt+jkj2D61X/nf3syf+EW8Pf9AHS//AOP/Cj/AIRbw9/0AdL/APAOP/Ctaijkj2D61X/nf3syf+EW8Pf9AHS//AOP/Cj/AIRbw9/0AdL/APAOP/Cvlqf44/EGa4llTWo4EdyyxR2cJVAT90bkJwOnJJ9Sa+h/hX4l1fxZ4Ds9V1m3jjuGd4lmQjFyqHb5m0fcJYMCPVSQACADkj2D61X/AJ397JPh/FHBD4ihhjWOKPXLlURBhVA2gAAdBXYVyXgT/mZf+w9df+y11tRR+BHRmbvipt+X5IKKKK1OAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA5L4Z/8AJPdL/wC2v/o168N/aO/5KHp//YKj/wDRste5fDP/AJJ7pf8A21/9GvXzz8edSmvvipeW8qxhLC3ht4ioOSpQS5bnrukYcY4A+pyofwo+iO7NP9+rf45fmyn8Ev8Akr2hf9vH/pPJX0l/zV7/ALgP/tevm34Jf8le0L/t4/8ASeSvpL/mr3/cB/8Aa9FXp6hgf+Xn+B/odbRRRWpwnJfEz/knuqf9sv8A0aleU/tG+FIYX0/xXbiNHmcWV0owC7bS0b8Dk7VZSSegQAcGvVviZ/yT3VP+2X/o1Ku+OPDf/CXeC9U0MSeXJdRfunLYAkUh03HB+Xcq5wM4zjmsl/Ffov1O6p/uNP8Axz/KmeOfs2a+wuNZ8OSNIUZBfQKFXapBCSEnrk5iwOR8p6d9P9pTU/K8PaHpPk5+03b3Pm7vu+Um3bjHOfOznPG3vnjxTwH4rm8GeMbDWUMhgR9l1Gmf3kLcOMZAJx8wBONyqT0r2fSJLH4yfF671C4hku/C+g26raRyHakkrNwzxkkkMRIegyI0DDqDqcJyn7PPiT+zPGlxobx7o9Xi+VwuSskQZxk54UqZM8E529Bmvd/iF4s/4QrwXfazGkEl0m2O2hmfaJJGIA92wMsVHJCnkdR8seKbF/h/8VbuOyjg/wCJZqCXVpGWZ0CZWWJWJwxwpUHnseT1r0TxF4iu/jv4l07wx4dEllo9un2y7kvIkDowJUvgMdwCuAqgjJc54AKgHT/s5aUtr4K1DU2tpI5729KCVtwEsUajbtzwQGaUZHfIPTjnP2htD8Qah4m0q6tLC+utMFosKGFGkRJ2lII2jO1m3RAZA3cAZxx9BwQQ2tvFb28UcMESBI441CqigYAAHAAHGK8n+J3xqXwbqkmhaPYx3eqRorTS3G4RQFtrBdowXJU54IAyvJ5AAO0+HXh648K/D/R9HvGzdQxF5hgfI7s0jJwSDtLFcg84z3r5k+M8ENt8W9eSCKOJC8TlUUKCzQozHjuWJJPckmvS/DGn/F/x5ocN3f8Aiz+yNJvOVdbdI7lkHKugjRSFY4H31yMnBBG7yD4iaPeaD471LTL/AFafVrqHyt97PnfLmJGGcsx4BA6npQB9f+E7641Pwbod/eSeZdXWn280z7QNztGpY4HAySeleEftG6Hpem6jod/Y2EFtdX32k3Twpt84qUIZgOC2XbLdTnknAx9B6TNY3OjWM+liMafJbxvaiOPYoiKgphcDaNuOMDFfOn7SN9cSeMtJsGkzaw6f50abR8rvI4Y568iNPy9zQBr/AAR+HXh3X/CR1rXdE+03S6g32aSZpAjxqqY+XIV137wcgg4IPTFeyXb6D4F8NX9/FZWmn6fbI1xLHaxJCHbAGAPlBdsKoz1OBXN/BL/kkOhf9vH/AKUSVofFSXyvhf4hb+zvt+bQr5O3dtyQPM6H/V58zPbZnI6gA+XPDlhffEn4kWdrql9JLcalcF7q4dsMUVSz7eCAdikKMYHyjgV9l2NhZ6ZZx2dhaQWlrHnZDBGI0XJJOFHAyST+NfMH7PN9b2nxKkhnk2SXenywwDaTvcMkhHHT5UY8+nrivqegD5E+MFovh/4qXK6Vp0ejxwJBLataI0IY7FPmrg4BDZGUwMp/eyT9L+JYb65+GmrwXptBqEmjzJMY5NkAlMJDYZyNqbs8sRgda1NT0LR9b8r+1tKsb/yc+X9rt0l2ZxnG4HGcDp6Cs/x3/wAk88S/9gq6/wDRTUAfHng7QG8U+MdK0VVkKXVwqy+WyqyxD5pGBbjIQMe/ToelfYdj4G8Lafoceiw6DYvpqSmcQTxCYGQ5G8l8ktg4yeQOOnFfMHwS/wCSvaF/28f+k8lfX9AHxp8StLtPCXxP1Gy0FJLGCyeB7by5XLRN5SPkMSWzuJOc8V9JaTqUOs/EWx1S3WRYL3wzHcRrIAGCvKGAOCRnB9TXyz47/wCSh+Jf+wrdf+jWr6S8Cf8AIb8Nf9iZa/8AoS1lV6ep3YH/AJef4H+h6bRRRWpwhXz/AK94G0XVvgYmvGK0s9WsklnF4SIvOAlKmNzj5yVAVAed20AgEg/QFeH+JZ4Yf2YnSWWNHmdUiVmALt9r3YX1O1WOB2BPasn/ABV6P9Dup/7jU/xw/KoeYfB/wdpfjbxlLYav55tbe0N1shfZ5hWSMbWOM7SGOcYPoRX0v4e+HXhHwreG80fRIILo9JnZ5XTgj5WckrkMQduM9814Z+zfBM3jrU7hYpDAmmMjyBTtVmljKgnoCQrEDvtPpX03Wpwnj/7R3/JPNP8A+wrH/wCipa4D4L/C3TvGMVzrmuN52m28rWyWaOyGSTaCWZhghQHGADknrgDDdX+0tps0ujaBqitH5FvcS27qSdxaRVZSOMYxE2ee469tT9nH/knmof8AYVk/9FRUAdZrfwn8Ea5b+VN4ftLV1R1jlsEFuyFh975MBiMAjcGA9OTnwz9n7xDcaZ8QP7HVd9rq0TJIMgbXjVpFfpk4AdcZH3884FfRfjSea18C+Ibi3lkhni0y5eOSNirIwiYggjkEHnNfPn7OP/JQ9Q/7BUn/AKNioA7/AON3w3s9b0O78U2EPl6vYxeZPswBcwr97dkj5kUEhupC7cH5dvhHw5TS5fiLoUOs2f2yxmu1iaAjIZ2+WPcMjKhypIPUA8HoftevjDx14Yv/AAT451KO1gntbW2u1lsriHzNsaOWeHbIedwCsM5+9G+CdpNAH2Pf2NvqenXNheR+Za3UTwzJuI3IwIYZHIyCelfIFr4Ht/EvxQn8MeGpL6OwjlZGuNQhPmQogxI7qAMfMCFDBeWUHaSce3+LviU8fwNtPEVm+L/V4ktEkgVkWGdlYSkZYMu3ZKFIJ+YKeRzVf4A+Df7F8LyeIbyHbfar/qd64aO3B+XGVBG85bgkMojNAHV/Daxt9M07XLCzj8u1tdZnhhTcTtRQgUZPJwAOtdrXJeBP+Zl/7D11/wCy11tZUfgR3Zl/vUvl+SCuK8XX1vpnjLwxf3knl2trFqE0z7SdqLACxwOTgA9K7WuS8Qf8lC8Hf9vv/ooUVvh+a/NBl38Z/wCCp/6RI+QfEOvX3iPWZ9Rv7u7uXd28v7VN5jRoWLBAQAABuPCqo5OAOlfVfh74N+DtE0M2FzpcGp3E0Xl3N3dJl3POSnP7r7xxsweBkkjNfNktg3w8+KUFvqQklTR9Thmdo1XdLErq6sBuwCyYOCeM4Jr7H0rVbHXNLt9T0y5jubO4TfFKnRh/MEHIIPIIIOCK1OE+SPij4StPAfxBNpaQxzabKiXlvbSyO2IySDG5G1sbkcDBztx82cmve/Bfwg8J+GNYtfEek3l9fSCIm2kmuEePDrjeuxV3ZUkDJIw2cZwR4p8T9Xb4lfFJLTw7ZyXTxINOt/LdX+0lHdjICOAnzE5zjau4kcgfWdAHnfxxMw+EmriKONkLwCUs5UqvnJyowdx3bRg44JOeMHg/2Zf+Zp/7dP8A2tXQftHf8k80/wD7Csf/AKKlrn/2Zf8Amaf+3T/2tQB0Hxx+Haa/o9z4qtZp/wC0tNtADBlfLeBGZnPOCGAdmzk5C4Ayc1wH7OP/ACUPUP8AsFSf+jYq9/8AHf8AyTzxL/2Crr/0U1eAfs4/8lD1D/sFSf8Ao2KgD6D8XeKLHwd4au9Yv5IwIkIhiZ9pnlwdsa8E5JHXBwMk8A18saJpHiv4y+Md93eSTuiILq+lQCO2iHQBVwMnnCDG45PHzMPY/wBo7/knmn/9hWP/ANFS1T/ZsnmbwrrNu0toYEvQ6Rqx89WZAGLjoEIVQp7lX9KANz/hQPgb+x/sX2e++0f8/wD9qPnfez0x5fT5fudPfmvAPiV4I/4QLxa2lR3X2m1liFzbOww4jZmUK/GNwKkZHB4PGcD7Pr5//aa/5lb/ALe//aNAHqnwxZm+GXh0vfx3x+xIPNRVAUY4j+XjKD5CepKHPOa+fPjT4At/BuuQX9rfz3EOsSzyiKfLPEV2FsyEkvkueTzxyWPNe3/BL/kkOhf9vH/pRJXnH7S/2z+0fDu/yPsPlT+Vtz5nmZTfu7bceXjHOd2e1AHd/AbUob74V2dvEsgewuJreUsBgsXMuV56bZFHOOQfqfNPjd8NbzSby78Yw6hPf2t5d/6RHKhZ7XcPl+YcGMEbRnbt+Rfm616P8AdM+wfC+C587zP7Qu5rnbtx5eCItuc8/wCqznj72O2TY+Ot9b2nwo1KGeTZJdywQwDaTvcSLIRx0+VGPPp64oA+fPht4V1LxxrN34dtNZk06zlt/tF5gsyyLGwCgxggOQzjGSMcnrwfU9T/AGa7M6dF/ZPiCdb5Ij5n2uENHNJgYxtwY1znrvIBHXHNf9mX/maf+3T/ANrV75PPDa28txcSxwwRIXkkkYKqKBkkk8AAc5oA+dPgj8Tb+31O08H6hHPfW91Li2uDJJI9sBHgIFw37v5FxjaEyxJx0+g9W1KHRtGvtUuFkaCyt5LiRYwCxVFLEDJAzgeor5M+CX/JXtC/7eP/AEnkr6n8WWNxqfg3XLCzj8y6utPuIYU3AbnaNgoyeBkkdaAPkzQNJ8R/FzxqtvdanJPcMhluLq5fcIIQ3O1cjjc+Ai4GW7DJHreu/s9+G7Hw9qd5p11rlxfQWkstvD5kb+ZIqEqu0R5OSAMDk1xn7Os8MPxGuklljR5tMlSJWYAu3mRthfU7VY4HYE9q+m7+b7Pp1zP9pgtfLid/PuBmOLAJ3OMr8o6nkcDqOtAHw54c0ObxL4gs9Gt7q0tp7tykcl3IUj3YJAJAJySNoGOSQO9fUfwv+E9v8P8Azr+5vPtur3MQid0BWOFDtLIoz83zD7xxkAYC858I+Clh9u+K+j7rT7RDB5s8mY96x7Y22ufTDlMHs23vivr+gArkvhn/AMk90v8A7a/+jXrra5L4Z/8AJPdL/wC2v/o16yf8Vej/AEO6n/uNT/HD8qh1tFFFanCZPin/AJFDWv8Arwn/APRbV8h+OvBl94Xn0+/nkjms9YgF3bSrwQSFZ0Zc5BUuOehBB65A+vPFP/Ioa1/14T/+i2ry/wCJfhH/AISP4L6PqVuub7RrCK5Tn70JiXzV5YAcAPnBP7vA+9WX/L35Hd/zA/8Ab/6HPfs0Wu/UfEV59onXyooIvJV8RvvLncy92GzAPYM3rWn+0H4M1rVUtfEljJJc2FhbmO4s1yTANxYzAZwQRgNgZARScgEr558EfE7+H/iLaWsk/l2Oq/6JMp3EFz/qiAP4t+FBIOA7dM5Ho/7Rl9qhs/D2iWEk7Q6lLKJbWFcm4dDF5a4HLcucL3ODjIGNThOI+Gfw3t/FNnBqek+PP7L1uLzDJaQQETwAHbuBEisVKsvzAY+bbnINfRfjDSdS1zwnf6do+pyabqEqKYLpHZCjKwbG5TkBsbSR0DHg9D8qeAvFF58L/HJudT0qdcxfZry2mjMcyRuVfcobGG4UgHgjjjOR9F/Fnxy3gfwc89nLGurXj+RZhgrbT1aTaTyFXvgjcyZBBoA8Y8C+GvHWu+MbjRofFd3aQeGLjY1wbh5ooJE3xoI4mIDAhXXBAGzcD12n2P42/wDJIdd/7d//AEojqP4J+HpvD3w3tku7a7try7uJbi4gukKNG27ywApAIBWNDz6k9CKufGB3j+FGvmO8+yMYkBkyw3AyICnygn5wSnp83JAyaAPmDwB4i8QeGvFCXXhq0+2380Twm1+ztN5yEbiNq/NxtDcEfd9Mg+iSfB74pTPDrsviCN9Wht/3W7U5jdJ8pPlLJjAPzMOH25J5wc1n/s6wQzfEa6eWKN3h0yV4mZQSjeZGuV9DtZhkdiR3r6joA+bPhh8WfEem+KrXwx4me7voLi4Fmv2kf6RazM5HzM3zMNzYYMcgAYxt2n3vxTb67d+HLuDw1ewWWrts+zzzgFEw6lsgq3Vdw6Hr+NfJGuxed8ZNTi/tH+zd/iCVft27b9mzcH95nIxt+9nI6dRX2fQB8cXXxG+IWja5PFc+Ir6O+s7ucSxPIsiLKfldSvKFQV4XBVeqgZzX2PXxB47/AOSh+Jf+wrdf+jWr7foA8P8A2lNT8rw9oek+Tn7Tdvc+bu+75SbduMc587Oc8be+ePQPhX/Z3/Cr/D39l/8AHv8AZBv+9/rsnzvvc/6zf7enGK8v/aa/5lb/ALe//aNdp8Brya5+FdnDLaSQJa3E0UUjZxOpcvvXgcbnZOM8oeewANT4ifE3S/h9ZxCWP7bqc+DDYpJsJTOC7Ng7V4IHByeAOGI8QsfGfxf8d3kb6NLfPCt2ZIzZ26QwRuoLeW0hABUAj5JGOflzkkVxev8Ai6bxT41bX9ahku7c3Ab7E05ULbhsiBXUDaNuRuAByS3Umvd4P2h/BNrbxW9vpGswwRIEjjjtoVVFAwAAJMAAcYoA88sfjT4+8PeI408StPcxwZ+0abPbR2rtuQ7ckR7l6q3TnHoa9u+F2pQ6zo2r6pbrIsF7q01xGsgAYK6owBwSM4Pqa8J+LvxB8N+Pv7Mn0nTb6C+tt6ST3KxpujOCFwpYtg5IyQFy3B3ZHp3wC1KaXRr/AEtlj8i3itrhGAO4tIrqwPOMYiXHHc9e2U/jj8zuw/8Autb/ALd/M9hooorU4TkviZ/yT3VP+2X/AKNSvmHxHrnj7wt4gvNF1LxTrIu7Vwr+XqcrKwIDKwO7oVIPODzyAeK+nviZ/wAk91T/ALZf+jUrzz9onwj9s0e08V2y/vrHFtd89YWb5G5b+F2xgAk+Zk8LWS/iv0X6ndU/3Gn/AI5/lTND4CXXinU9DvtR1jW/t+myS7LeOeYzzpIuNxLFiUXGPkbJP3htBy/OfFvxL8UPCuqXLnU47fQb93jtHsYVxGvICM5XekpUbsg9Sdp+UgV/2adShi1nX9LZZPPuLeK4RgBtCxsysDznOZVxx2PTvv8A7Rt5NLo3h7Qbe0knnvr1pY/LyzFkUIECgZYsZv06HPGpwnIfCLWfiR4g8Sqtlrl3c6XDcQNqbX1wJgsWWO1fM3MCwVx8nfGSMAj3Px//AMJd/wAIu/8Awhfkf2t5qZ83Zu8vPzbN/wAm7OPvcbd2OcV84fB7xJeeEPiVb6dcxzxw6hKNPu7ZlIZJC21CVJGGV+CTkhWfAya9z+L/AMQJvAnhqEaf5f8Aa2oO0VsXUkRKo+eTGMErlQAe7A4IBBAPKPAXxK+JPinxrpempqcl3bm4je8RbOBQtuGHmFmCDaNvGcg5IA5IrQ+JXxP+JHh7xLcWAij0ezjuJltJ0swwu4sgq2+TerEKVztxgsQRngd/8EvBM3hLwc11qFvJBqmqOJZ45AVaKNciNCMkZwWboCN+CPlrL/aO/wCSeaf/ANhWP/0VLQAfAvxt4i8Y/wBvf2/qH2z7L9n8n9zHHt3eZu+4oznavX0qn8dfFniaysLrQ9P0S7i0ea3X7Xq2xipDOBsV0OEBxsYPy27GAMFsv9mX/maf+3T/ANrV6R8YPsf/AAqjX/t3n+T5SbfIxu8zzE8vOf4d+3d325xzigD5M8OTa1beILOfw6Ls6tG5e3FpGXkJAJOFAO4bc5GCCM54zX0n4A8dfETXvFCWPiHwf9h01onZ7r7JNbeUQODmRiHyfl2jnnPRTXnH7OP/ACUPUP8AsFSf+jYq+n6APn/9pr/mVv8At7/9o1554Q+K2teCfCt9ouk29pvubgzpdSqWaIsmxsLnBPyoRngYOQ2ePQ/2mv8AmVv+3v8A9o1Y/Zz8MaXLpN74nlg83U47t7SF35EKCNCSo7Md5BPoMDGWyAcRqfxP+LeieV/a13fWHnZ8v7XpUUW/GM43RDOMjp6ivVfC/wAUYfiD4I8SW9xaR2WqWemyNJGsoZZlMZBdAfmADcEc7dy/Mc10XxeTSz8L9al1az+0xxRAwYGWjnYhI3ByMYZxnB+7uGCCQfk3w7qt9pOro9hcyQG4U2s23pJE/wArIw6EEfkQCMEA1M/hZvhf48PVfmfanhb/AJFDRf8Arwg/9FrWtWT4W/5FDRf+vCD/ANFrWtRD4UGK/jz9X+YV5V8ZdY1TQG0bVNFTff28V4Yz5XmbAUQM+3/ZUs3PA25IIzXqtcl4g/5KF4O/7ff/AEUKit8PzX5o3y7+M/8ABU/9IkfNM3xn+IM6BH8RSAB1f5LaFDlWDDlUBxkcjoRkHIJFfQeh6t42uPg5ZatFbQaj4pniWaOOfy0SVHlypO1kUfuSD1HvzkV8yePPCk3gzxjf6M4kMCPvtZHz+8hblDnABOPlJAxuVgOlfTfwY8Sf8JH8NbDdHsm03/iXyYXCt5artI5OfkKZPHzbuMYrU4Twj/hdXxH0/UfLvNVzJby4mtbixiTJU/MjgIGHQg4II9Qa9T+Cfj/xX4tuL+116GS7tEQyRamLcRqrgoDCSqhCcMGHQjB6gjHmnx+1P7f8UJ7byfL/ALPtIbbduz5mQZd2Mcf63GOfu574HX/s3eJP+Qt4XeP/AKiEUir/ALkbhjn/AK54AH97J6UAXPjJ8S/GnhTxBHpmlwR6bYOiyQX5iWVrnA+cDcCqgFgCuN3yg5wwFanwc+I/inx1qN3barbWLWNhaL5l3ChSRpiQF3Ddj5gshO1QAR24Bw/2itcmurjRfB9nayTTyut6QsZZnYlookTByST5mRj+7g9RXrngvwnZ+CvC9rotm/m+Vl5pygRppGOWYgfgBnJChRk4zQBwfxW+MU3gnVING0W2tLrUNgluXuCWSJTnam1WB3nhuSMArwd3Hlmp/E/4t6J5X9rXd9Yedny/telRRb8YzjdEM4yOnqK6SH4YeI2+P5vJ7SRNM/tNtXF8i7ovL8wyKmTj5y2EK9RktgqMn2/xRqvhnTtLeLxRc6alnKjP5F9tYTBMMdsbZLkHacAE5x3xQBx/wz+L9j46c6ZfQR6frSIGWIPlLkBfmaPPIIOTsOSByC2Gx2HjDxLD4P8ACd/r09vJcJaIpEKEAuzMEUZPQbmGTzgZ4PSvjzwJ/wAlD8Nf9hW1/wDRq19v0AfBmralNrOs32qXCxrPe3ElxIsYIUM7FiBkk4yfU16Z4R+O+seF/C66LLpkGofZojHZTyTOrR8sR5g53qMqAo2YVcZ7ji/iBBptr8Qdft9Jikhs4r2RFjdVXYwOHCheAgbcFH93bnmvsfw1ps2jeFdI0u4aNp7Kyht5GjJKlkQKSMgHGR6CgCPRNZuL3wvBrGtaf/YkjRNNPb3EwP2dASQXYgY+UBjkDbnB5BrxTxd+0VMXu7Lwrp8aoHKRalcksWXaRuWLA2ndgjcTwOV5wH/tE+Mv+PTwhZTel1f7G/79xnDfVyrD/nmRWB8HfFfw98MZudctp7XXU8zbqMqtNHsO0BUVASjYzztPG758NtABTk+L/wAUPDmqQrrUkgfZv+x6jpqwiRTkA4VUfGQcEEcr35Fe9+BfiRoXjyzH2GbydSjiElzYSZ3xc4ODgB1z/EP7y5Ck4rl/E/xG+E/i7Q5tI1fWvNt5OVZbOcPE46Oh8vhhk/mQQQSD4Z8J9bm0P4m6HNF5jJc3C2csayFA6ynZ83qFYq+D1Kjp1AB6vqv7Sdja6pcQaZ4ekvrON9sVy935JlH97Z5ZIGc4yc4xkA8DQs/2ivDk3h+6u7rT7u31SFMx2Gd6zsSQAsoGAAMFiwGM8Bsc0/2jND0uPwzZa3HYQR6m+oJDJdIm15EMT8MR97/VpjOcYwMZNeafBjwlpHjHxrNZa1DJPaQWUlx5KyFA7BkQBiuDj5yeCOQO2QQDrNG/aR1hNRT+3NGsZbE4DfYQ8cicjLDezBsDPy/Lk4+YVHqX7SevS3Ctpeh6bbQbAGS6Z52LZPIZSgAxjjHY888ehw/Abwdb+KLfVoopzZxZY6ZK/mQs4C7TlvmKjDEqSdxI6KCp4/48/DyzstHs/EOgaTY2VvaZivxbKIsqzKI22DCnDFgSPm+ZeoHygHQfDv4x6p478aS6T/wj8ENiYjMJEucvbIowxbIHmbnKAbQuN3OcZrc+LPxD1L4f6XYzadpUd0947p9pnLeVAy7SAwXG4spfA3L90nnBFeSfs8+JP7M8aXGhvHuj1eL5XC5KyRBnGTnhSpkzwTnb0Ga9L+Pmv2Om/DybR52k+2aq6LbIq5GI5Ed2Y9AAMD1yw4xkgA8btPit4w1W4WXVNSjvU04i+hikto0XzYz8pOxVJHJGM96+uK+V/hZ4UmhTSvFdwJESbW7aytVOQHXcWkfkcjcqqCD1DgjgV9UVlD45fI7q/wDutH/t78wooorU4TkvD/8AyULxj/25f+ijXyb47/5KH4l/7Ct1/wCjWr6y8P8A/JQvGP8A25f+ijXyz8TtNm0r4m+Iredo2d717gFCSNsp81RyBztcA++evWsqPw/N/mzuzH+Mv8FP/wBIifQfgfxJ/wAIj8BNL1zXY76SO1i5QLmUxtOUi2hyPl2smOcbcY4xXH/8NNf9Sj/5Uv8A7VXoHwssNH1P4OaJZm0+12EkTebDfRpIGkErF/l6FRIGK55wFzzXzh8UPDeneFPiBqOk6VJm0TZIkRZmaDeobYSQM4zkYLfKVyS2a1OE+s/B/iWHxh4TsNegt5LdLtGJhcglGVijDI6jcpweMjHA6Vj+PviZovgC3jW8El1qE6M0FlCRuIAOGcn7iFhjPJ64BwcU/gl/ySHQv+3j/wBKJK8M+POpTX3xUvLeVYwlhbw28RUHJUoJctz13SMOMcAfUgFyb9obxrJeW06RaVFHFu3wJbsUmyMDcS5YY6jaV98jirHhv9obxLpnmJrlrBrUbZKt8ttIp4wMou0qMHjbnJ64GK9j+DmgL4f+GWlLtj8++T7dMyMzBjIAUPPQiPywQOMg9ep8U/aGsbe0+JUc0EeyS70+Kac7id7hnjB56fKijj09c0AZcfxd8bz+NZtU02/u3N3cbYdLYmeLYWG2JY8AZwFXcoVjzzljXt/xXvJtR+BF/e3FpJZz3FvaSyW0md0LNNEShyAcgnHQdOgrH/Z1tLF/BV1frp1pHfpey2zXap+9kTbG+GYknGSOBgfKDjOSek+Nv/JIdd/7d/8A0ojoA+YPBPif/hDvF9jr/wBj+2fZfM/ceb5e7dGyfewcY3Z6dq9nvv2l7OO8kWw8MTz2oxsknvBE7cDOVCMBzn+I+vHSvMPhl8Pv+Fga5JbSalBaWtrtkuU3fv5IznPlLjBwQAWPC71OD0P0HB8Dvh9DbxRPosk7ogVpZLyYM5A+8drgZPXgAegFAEfgX4zeH/GUosrgf2TqZwFt7mVSkpLbQsb8bm5X5SAfm4BwTXUeLPGmheCtOW81q78rzdwghRS8kzKMkKo/AZOFBIyRkV8WTzw2usy3GjS3cMEVwXs5JGCzoobKEleA4GDkd+lfZfjrwLpfjzQzYX48q4jy1rdouXt3PceqnAyvfHYgEAHxRXsHgn46f8Id4QsdA/4Rz7Z9l8z9/wDbvL3bpGf7vlnGN2OvavH6+g/DfwA0HWfBWm6lPq+pJqF9ZJcBk2eVG0i7lGwrkhcgH5hnB+7ngA7f4SeIbfXtO1mZV8q6nvjfyW+S3lJMAV+bAB5Rx/wHoMivRa86+Enh630HTtZhVvNuoL42Elxgr5qQgBflyQOXc/8AAupwK9FrKj8CO7Mv96l8vyQUUUVqcIUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQB81zfGi/wDB+i6Toeh2ljcSQRSPdvdpI212lcqgAK9Fwc5Od4HBBz5h4t8VX3jPxBLrOow2kVxIioVtYti4UYGepY+7EnoOgAH0zJ8DvCc0heV753PVmMRJ/wDIdM/4UV4P/wCnv/yD/wDG6wp+0jBR5dvM9bFrCV8ROsqtuZt/C+rufNPhLxVfeDPEEWs6dDaS3EaMgW6i3rhhg46FT7qQeo6Eg/Q3w48df8LA8eXGrf2d9g8nTGtvK8/zc4lRt2dq/wB/GMdq0P8AhRXg/wD6e/8AyD/8brf8KfDrRfB1/LeaW9zvkiMTJIU24JU5wqjn5RQ+eTWnUin9Woxm1Uu3Fq1mtzrqKKK3PMOS+Jn/ACT3VP8Atl/6NSutrJ8TaJ/wkfh660r7R9n8/Z+92b9u1w3TIz0x1rI/4R/xf/0PH/lJi/xrF80ajaV9F28/M9KCpVcLGnKootSk9VLZqNtovszwn9oDwpDofjGDWbURpBrCM7xrgbZk2hzgADDBkbOSSxcntXs/wg8KTeEvh9aW92JFvL1ze3Eb5HlM4UBMEAghVQEHOG3c4xV//hH/ABf/ANDx/wCUmL/Gj/hH/F//AEPH/lJi/wAaftJfyv8AD/Mz+qUf+f8AD7p//IHlP7S2lKtxoGsR20m90ltZ7gbiuFKtGh7A/NKR3PPXHG/+z14Uh07wnL4klEb3eqOyRMMExwxsVx0yCzhiQCQQqdxXX33hHxHqdnJZ3/i2C7tZMb4Z9FgkRsEEZUnBwQD+FSQeGPFNrbxW9v4yjhgiQJHHHo8KqigYAABwABxij2kv5X+H+YfVKP8Az/h90/8A5A7GviTx/pF9oXjzWbDU7yO9vBcGaW5RNglMgEm7b0UkOMgcA5AJHNfV3/CP+L/+h4/8pMX+Nc34l+ENx4vuILjXfEUd1PAhSOQacsbbSc4JRwSM5IBzjJx1NHtJfyv8P8w+qUf+f8Pun/8AIEFh8fvCP/CL217qNxP/AGt5SfaLC3tX3eZkBthJ2bc5YZfO334r508YeJZvGHiy/wBent47d7t1IhQkhFVQijJ6naoyeMnPA6V79Y/s/wBhp95HdQ6pA8iZwJ9PEyHII5R3Knr3HHXrWx4k+FF14u8s654hgu5I8BZv7KjjkAGcLvRg235iducZOcZo9pL+V/h/mH1Sj/z/AIfdP/5AzPg18Tf+Em+yeFP7I+zf2ZpSf6V9p3+b5flx/c2DGd2epxjHNcp+0tpsMWs6BqitJ59xby27qSNoWNlZSOM5zK2eew6d/RvD3w11PwpZm10PxLBZRt98ppEReTBJG5ySzY3HGScZwOK0L7wj4j1Ozks7/wAWwXdrJjfDPosEiNggjKk4OCAfwo9pL+V/h/mH1Sj/AM/4fdP/AOQPGfhJ8X7Hwdo0uha5BdvaK8txBcRv5mw7QRCIz0DMGOc43PyACWHufh3xJoXxL8JXU9rHO9hcebZXME6mNxlcMpKnurA5U/xdQcged/8ADOWl/wDQX/8AJZv/AI7Xb2PhHxHplnHZ2Hi2C0tY87IYNFgjRckk4UHAyST+NHtJfyv8P8w+qUf+f8Pun/8AIHyjqtlfeCfGtxaxSyJeaTe5gmeHaWKNmOTY2RggKwByCCOor3ux/aR8NyWcbX+jarBdHO+OARyovJxhiyk8Y/hHpz1rf8S/C/UPGFvBBr3iaO8SBy8ROmIjISMHDKwODxkZwcD0Fc5/wzlpf/QX/wDJZv8A47R7SX8r/D/MPqlH/n/D7p//ACBia98Zdd8daxZ+GvA1vPpv22XyTdy4MzKy4JwobylXLMWUlgFDArgg9v8AFz4kaFoHh7VvD3nfatZvLRrf7LFn9ysqEb3bGBgHO37xyvAB3B+ifCefw55DaTrVjbzQbvLuP7EheZd2c/vWJc8Ejk9OOlZ+t/A5PEesT6tq2v8A2i+n2+ZL9j2btqhRwsgA4AHAo9pL+V/h/mH1Sj/z/h90/wD5A8N+Fut6d4c+I+k6tq1x9nsYPO8yXYz7d0LqOFBJ5IHAr6//ALd0f+x/7X/tWx/sz/n9+0J5P3tv387fvcdevFeR/wDDOWl/9Bf/AMlm/wDjtb//AAqm+/4RD/hFP+En/wCJJ/z6/YF/56eZ9/fu+/z19ulHtJfyv8P8w+qUf+f8Pun/APIHzT4yv7HVfGut6hphkNnc3ss0TO2S4Zid33VIBJJAIyAQCSRk/RXwz1Wx1bV9BewuY5xb+FIbWbb1jlSRVZGHUEH8wQRkEGsz/hnLS/8AoL/+Szf/AB2ut8BfC+38CalPd2+o+eksLRmPyCuCSpzku39zGKmTlJpcrWvl/mbUoUaEZy9tGV4tJJTvd+sUvxO/ooorc8sK8W1qCa5/ZiukgiklcJvKopYhVu9zHjsFBJPYAmvaa5L4Z/8AJPdL/wC2v/o16yf8Vej/AEO6n/uNT/HD8qh80/Cr4gTeBPEoMvlnSb944r4MpyignEikAnK7mOBncCRjOCPp/QPH3hfxTerZ6Lq0d3cG3Nz5axupWMPsJbco2ndj5Tg4IOMEGuH8Z/ATRfEeqJf6Pdx6ESm2aCC0DxORgBlQMoQ4644PBwDknpPBHwq8P+AtRub/AEyW+nup4vJL3cqtsTIJACqo5IXrn7oxjnOpwnmH7RupaXqX9hx2WrWNxdWUtzFPawzb5IydmdwXIXBQghiDk8A4ONP9mzW4X0bWdBby1nhuBeJmQbpFdQjYXrhTGuTz98dO+p/wzj4P/wCglrn/AH/h/wDjVaGh/AfwnoOuWWrRXOq3E1nKs0cc86bN68qTtRTwcHr25yMigDc+KPiPSNC8C6rb6lfRwT6jZXFtaRkFmlkMTAAAAnGSAWPAyMkZFeAfAjVbbTPibbpdXMkAvbeS1jxsCO5IZUctyASvG3ktsHQkH3vxv8KvD/j3Uba/1OW+guoIvJD2kqrvTJIBDKw4Jbpj7xznjHL/APDOPg//AKCWuf8Af+H/AONUAewV4n+0V4Xa+8P2PiS3jj36c5huiEUMYpCApLZyQr8BcH/Wk8YOfbK4f4seJ9L8O+AdUhv59txqVpNaWsCcvI7oVyB/dXcCT2+pAIB8ueDPDWpeN/EFl4ctbiSOAu8zuwZ47dcDfJtHAJCovbJ2AkcV9p2Fjb6Zp1tYWcfl2trEkMKbidqKAFGTycADrXlfwH8DN4e8NPr1/FGL/VkR4cFWMdtgMvOMgsTuIyRgJnBBFeuUAcl4E/5mX/sPXX/stdbXJeBP+Zl/7D11/wCy11tZUfgR3Zl/vUvl+SCuS8Qf8lC8Hf8Ab7/6KFdbXJeIP+SheDv+33/0UKK3w/Nfmgy7+M/8FT/0iR5Z8X9Yt/HPjnRPAWlJBNJHdqLm/iiMslu7Eq6ADHyoo3vg9VAJUoauab+zTpsVwzap4ku7mDYQqWtssDBsjksxcEYzxjuOeOfOPik+keKPibOvguyku3lTE/2KIuLm4BYyPGq53DbjJAAJVm5zuNO1sPinY+f9jtPGVv58rTzeTHdJ5kjfedsdWOBknk1qcJ9R+GvAPhfwhcT3GhaTHazzoEkkMjyNtBzgF2JAzgkDGcDPQV0lfHE3ir4meFLy2ur/AFPxHZSNu8kakZSkmBhvklyrY3DscZB64r2f4P8AxY1TxtqMuh6vZwG6t7Q3H22E7PMCmNMMmMbiWJJBA7BRQAftHf8AJPNP/wCwrH/6Klrn/wBmX/maf+3T/wBrVc+IPwO17xN4lm1bTvEMdyk7sfJ1SR82yk7gkbKGym5nwMLtGBzyazPDHwX+InhHXIdX0jWdDiuI+GVp5ikqHqjjy+VOB+QIIIBAB6n8WJ9Ih+GWuJrEsaRTW7JbqzEF7jG6ILjkneqnHTAJPANeIfs6zww/Ea6SWWNHm0yVIlZgC7eZG2F9TtVjgdgT2rr/AIkfBPXvFnje813S9R01YLtIy0d0zoyMqBMDarAjCg546kY4yeU/4Zx8Yf8AQS0P/v8Azf8AxqgD1f44+HrjX/hrcvat+802UX5TA+dEVg/JIxhXZu+duAMmvLPgF43sfD2qajo+sX0dpZ3qLNDLcT7Io5Uzkc/KCyn7xI/1ajkkV9H6THfQ6NYxapNHPqCW8a3UsYwryhRvYcDgtk9B9BXifjr9n59T1w3/AITnsbK3ny09pcsyJE/rHtVvlPPy8be3BAUA94r5Q+PGuaXr3j63l0m/gvYYNPjhklgfem/fI2Aw4bh16E+nUEVsQ/Cv4ra/Z3Nlf3cGl2Hy4sXu1itn5z8sNuGjXBAY/KMk55OTVzV/2bNSjS0Gi65aXD7GFyb1WhG7d8pQKH42kAg91zk7sKAex/Di+t9Q+Gvhya1k8yNdPhhJ2kYeNRG459GVh7444ryf9pr/AJlb/t7/APaNdJ8I/hZr3gLWdQvdU1S0kguLcRLbWjuyu24He24KMqAQOD99uR388f8AZ88cX15dTXWo6U0zSktPNdSsZyQGL52EnkkHdg5B4xgkA9X+BV9b3fwo02GCTfJaSzwzjaRscyNIBz1+V1PHr65qn+0Fps198MjcRNGEsL2K4lDE5KkNFheOu6RTzjgH6HjPDHwK8caDrkN7b+KLHS8cPcWTyyPgfMFKFVV1LKuVY49jjFdP49+C+qeMtYu9TPjCd1+Z7Oxu4N0cBKgbAysAqkqOQmcYJ3HkgHP/ALMv/M0/9un/ALWr6Ar540P4A+LNFvLLVLXxNY2WpwXakNAruI4sfM4JA3NyR5ZUKw6tziu/+LnhLxZ4q060j8M6n9mjhiuBd2v2p4ftYYLtTAG1ujD5yB83XBNAHgnwYnhtvi3oLzyxxIXlQM7BQWaF1Uc9yxAA7kgV9h182Sfs2a8NLhki1zTW1AviWBldYlXnlZMEsfu8FB1PPHP0PAzabo0TapfxyvbW4N1eyKsKsVX55COiA4Jx0FAHyhpPgzV/iB4x1DVvBNhJpmlpetJDc3EoiW1bl1ClBkEHGFQNs3KCf4j6nffD/wCK3iezk0fxL4x0o6RcY+0CC2V3+UhlwBEmfmVf4h+PQ+AeHvFWu+FLw3Wh6nPZSN98IQUkwCBuQ5VsbjjIOM5HNdR/wu34h/8AQw/+SVv/APG6APpfwL4F0vwHoYsLAebcSYa6u3XD3DjufRRk4XtnuSSeor5kg/aQ8VLcRNcaVo0kAcGRI45UZlzyAxcgHHfBx6Gvofw5r9j4p8P2etaa0htLpCyeYu1lIJVlI9QwI4yOOCRzQATeJdBttUGlz63psWoF1QWj3SLKWbG0bCc5ORgY5yKxvhn/AMk90v8A7a/+jXr5h8W/CvxX4Mt5bzUbKOXT43VDe2sgePLDjI4dRn5csoGcDuM/RXwe1Ka+8HT28qxhLC9e3iKg5KlEly3PXdIw4xwB9Tk/4q9H+h3U/wDcan+OH5VD0GiiitThMnxT/wAihrX/AF4T/wDotqPC3/IoaL/14Qf+i1o8U/8AIoa1/wBeE/8A6Lajwt/yKGi/9eEH/otay/5e/I7v+YH/ALf/AEPjrx54Um8GeMb/AEZxIYEffayPn95C3KHOACcfKSBjcrAdK9r+F+rP8TfiLdeMNUtoLe40XT4bS3hh3Y3yeZukyW/66gKQeHHOV3G5+0B4R1rxHpej3+j2Ul6NPeVZoIFLy4k8sBlQDLAFOccjIOMZI7T4Y+Ef+EL8DWWmyrtvpf8ASb3nP75wMr94j5QFTIODtz3rU4TyD9pTTPK8Q6Hq3nZ+02j23lbfu+U+7dnPOfOxjHG3vnjn/Bi3nxX8aeHNG1mLfpGi6eInht5DGPJjGAxyx+Z2MSsVwSMYxjI93+LXha48XfD+6sLCz+16lHLFNaJ5ojw4YBjkkD7jSdfX1xWP8C/Cuo+GPBt1/a+mfYr68u/OG8L5jQ+WmwNjkYJf5WwQSeBmgD1CvL/j9dXlv8L54ra382G5u4Yrp9hbyowS4bI+786IuTx82OpFeoV84fFj/hYvjDxRNoEfhy+GjQXai0WG23xyMAVWZp8YGQ5yNwVRweVJoAz/ANnH/koeof8AYKk/9GxV9P18ceGNL+InhHXIdX0jw3rkVxHwytp0xSVD1Rxt5U4H5AgggEe7+Oda+J1l4L0q50bRoF1OWInVBZqLl7Z8ptESkndnLbgFfbzzxuIB8+eJdSm0b4uavqlusbT2WuzXEayAlSyTlgDgg4yPUV9p18ST+C/G11cS3Fx4a8QTTyuXkkksJmZ2JySSVySTzmvoNPF/xJi+FVrrA8L+br8d2IZ4Jrdy80ABHneSpVlYvtBUehYAKRgA+ePHf/JQ/Ev/AGFbr/0a1fb9fFF/4S8canqNzf3nhjXJLq6leaZ/7NlG52JLHAXAySelfR/w78S+PdW8Patc+JPD3l3VnEq2MbxG0kvZQjFg284XJ2fMFC5Y46EAA4f9pr/mVv8At7/9o13nwQsFsvhXpjpfSXQuXln5ZikJLlTGgYAgAryORuLkEgg14Z40k+Jnjy8gn1jwxqojt932eCDSZUSLcFDYJUsc7AfmJ9sDivQ/gtrXjLR3tPCWseFdSj0svI0N9PayQi1G13KsSmGDN0yQQWPJ4AAPBNCurOx8Q6Zeajb/AGixgu4pbiHYH8yNXBZdp4OQCMHg19t6l4a0HWbhbjVNE02+nVAiyXVqkrBck4BYE4ySce5r54+Ofw5m0bWZvFOl20jaXevvvCHL+RcMxySMZCMSMHJAYkcZUVJ4Y/aH1TR9DhsNX0n+17iH5Vu2u/Kd07B/kbcw5+bjPGcnJIB7Xq3hbwTo2jX2qXHhLRmgsreS4kWPTYSxVFLEDIAzgeornvg3r1nqunavBa2X2L/SVvhAgHlxRzAhI1Ix93ymHQDGMeg8Z8S/Enxf8VbiDw3Z2ccUFxcExWVnu3Tc5USuTghQMk4VeNxHAx7t8LbO4t9O1qTUjBNq51OSK9uokA811C55wPl3M5AwMbjwM1lP44/M7sP/ALrW/wC3fzO9ooorU4TkviZ/yT3VP+2X/o1K6HVtNh1nRr7S7hpFgvbeS3kaMgMFdSpIyCM4Poa574mf8k91T/tl/wCjUrrayX8V+i/U7qn+40/8c/ypnw5BJrXgLxjFK8Mlnq2l3AZopCRyP4TtI3IynHBwytwcGve/AGoWfxI+LmseMlM4tdJtIbbToJ4wjx+YrBmJRsHBE3B3Z8zttArjP2itAXT/ABjY61EsapqluVkwzFmliwpYg8AbGiAx/dPHc+1/DHwj/wAIX4GstNlXbfS/6Te85/fOBlfvEfKAqZBwdue9anCeCfGuO+8PfGV9YgmjSeVLa+tHUbjGUAQFgRjO6InHIxj3FaGvxN8ePiNaS+HLe7tbC2so4by6vI1AgAkkbOFYgkhsKuQSQegBI6/9pDRJrvw1pGsxeYyWFw8UqLGWAWUD52b+EBo1Xkclxz67HwH8JTeHfBT6jeQxpd6u6XClZCxNvtHlBh0B+Z245w4zyMAA9Urx/wDaO/5J5p//AGFY/wD0VLXsFfInxM+KGteNnGlXmnR6XZ2dwX+xkFpVlVdp8xmAOQS/AVcbsHJANAHd/sy/8zT/ANun/tavQPjb/wAkh13/ALd//SiOvnz4Z/Eyb4dXGosulx6hBfJGHQzGJlZC20hsMMYdsjHpyMc/V+rWDeIfCt9pziSyfUbKSBvMVXaAyIV5Ctglc9mwccHvQB86fs4/8lD1D/sFSf8Ao2Kvp+viCxutd+HfjSO4Nv8AZNX0yUhoZ0DDkEEH1VlY8g8hsg9DX0P8Kvib4g8Z/wBq3Wu6ZY2mkWUW86jDuijRxyyNvY5+U7iQRtA5+8KAOX/aa/5lb/t7/wDaNdB+zj/yTzUP+wrJ/wCioq8Q+I/jr/hYHiG31b+zvsHk2i23lef5ucO7bs7V/v4xjtXcfCT4u2fhmz0/wvf6TBFay3ZD6lHKIyu88PKCMNg4BbcMIBwdvIB6v8bf+SQ67/27/wDpRHXzT8O/+RvX/rwv/wD0kmr1r47/ABJh+z3/AIFtLOQzl4jeXEuAoXCSqI8HJJOMk4xgjBzkeI+G9Sm0rXre4gWNndZLchwSNsqNEx4I52uSPfHXpUz+Fm+F/jw9V+Z9qeFv+RQ0X/rwg/8ARa1rVg+Cb631DwTo81rJ5ka2qQk7SMPGPLcc+jKw98ccVvUQ+FBiv48/V/mFcl4g/wCSheDv+33/ANFCutrkvEH/ACULwd/2+/8AooVFb4fmvzRvl38Z/wCCp/6RI4j9oXwpDqPhOLxJEI0u9LdUlY4BkhkYLjpklXKkAkABn7mvGPhp4+m+H/iCa9MUlxZ3Fu8c9srkb2AJjPXAIbA3EHCs+ASa+w7+xt9T065sLyPzLW6ieGZNxG5GBDDI5GQT0r4s1TwZfab8QW8HtJH9rN6lpFLJ8qsJCPLc7S20FWVsckZx1FanCegfC/4Yr4/8P+INT1+O7SW7df7P1R3YsZcuZX2k/vBu2gk9fmAIIJHF/C/xPb+EfiBp2qX088NgN8V0YcnKMpA3KPvKG2sRz93IBIFfY9hY2+madbWFnH5draxJDCm4naigBRk8nAA618kfGfw3/wAI58Sr/bJvh1L/AImEeWyy+YzbgeBj5w+Bz8u3nOaANjwj4bl+NfxF1vWNXkntbBf3sxgZN65+WGIEjsqn5tpz5fOC2a+n76/s9Ms5Ly/u4LS1jxvmnkEaLkgDLHgZJA/GuX+GPhH/AIQvwNZabKu2+l/0m95z++cDK/eI+UBUyDg7c96k+JegX3ij4eavo+mLG15OkbRI7bQ5SRX256AkKQM4GSMkDmgDxjU/ip44+IXiWTS/AkN3Z2727p5C+UXZQWzK0jKPKJUqOG4OMMSRWuvwDsbfw1qOr+L/ABFd/wBqKk11PdWx82KMAFt7Bl3yngseVJzgep8c8M+JNY8A+KBqNnH5V9beZBNbXKuqnIKskigqeDg4PRlHpXoHi74reIPifo7eHND8NTxxviW7jtt13JIispXog2KGwSccnaMgZBAOT+Eumw6r8VPD9vO0iolwbgFCAd0SNKo5B43IAfbPTrX2XXwppeqav4S8QLe2TyWOqWTun7yIFomwUYFXBGcEjBHFfW/wu8Ua14x8HDWtbtLS3ea4dbb7KCFeJcLuILMQd4cc46DjuQD5g+J0ljL8TfETadDJDAL11ZXOSZQcSt1PBkDkexHA6D7Tr44+Lnh648PfErVknbfHfytfwPgDckrEngE4w25ecZ25wARW5pX7QHjLTNLt7F49NvjAmz7RdxSNK47birgE44zjJxk5OSQDP+OME0Pxb1d5YpESZIHiZlIDr5KLlfUblYZHcEdq7j4MfDnwf4s8FTahrGnSXt4l7JCzNLJEEAVCFXY+GGGzkgHLEYwATb8ffD3xL8RvDmi+KhYfZPEa2iwXWkvIqhk3sQyliNjfMWKMeAcZ3L83lngXx1q/w18QXBFtI0DPsv8AT5AImkZA6qCzKWQqzE4GOmDQB9F/8KS+Hn/Qvf8Ak7cf/HKjSX4X+EfHiWcNvptl4kvHiRI4LZmMbONiBdqlISwbkDbkMC2QQa4vxD+0jZrZhfDWjTvdN1k1IBUj5H8KMS+Ru/iXHB56UfCHwLqmua4/xC8YD7TNcfvbJblcu78bZ8cBVAGEGPQgKFQkA2P2jv8Aknmn/wDYVj/9FS1wH7OP/JQ9Q/7BUn/o2Ko/i58WbHxvZHQtM0+QWdterPFfPJgzBUZf9XtyoJckZOcAZAJwMP4S+N9H8B+Ib3UtWtb6fzrQ28ZtCh25dWOVYjOdo5DcYPBzlQD6/rH8V6N/wkPhLVtICQPJd2kkUXnjKLIVOxjwcYbacgZGMjmtSCeG6t4ri3ljmglQPHJGwZXUjIII4II5zUlAHwppN5N4a8VWN7cWknn6ZexyyW0mY23RuCUORlTlcdOPSu88caxb/Fr4tada6Kk6Wr+Vp8dx5RdmQOzNNs4IUB2OCfurk7eQLn7QHhSHQ/GMGs2ojSDWEZ3jXA2zJtDnAAGGDI2ckli5PatD9nrwY2oa3L4tnkkSDTnaC2VduJZWQh93OQFVx25Ljn5SCAeveLLG30zTvCFhZx+Xa2us2cMKbidqKGCjJ5OAB1rta5Lx3/zLX/Yetf8A2autrKHxy+R3V/8AdaP/AG9+YUUUVqcJyXh//koXjH/ty/8ARRr5N8d/8lD8S/8AYVuv/RrV9JT+NvDvg74heJv7f1D7H9q+y+T+5kk3bYvm+4pxjcvX1r5p8Y65D4l8Y6rrNvax20F3cM8caRhPl6AsASN5A3Mc8sWPesqPw/N/mzuzH+Mv8FP/ANIifW/wx02HSvhl4dt4GkZHskuCXIJ3SjzWHAHG5yB7Y69a+bPjb/yV7Xf+3f8A9J469r8FfGPwpN4Kt31bWJLa8063ggu/tuWlnfaqmRACzSAtnJHI6sACCfnz4ieJLPxd471LXLCOeO1uvK2JOoDjbEiHIBI6qe9anCfS/wAEv+SQ6F/28f8ApRJXkn7QHgxdG8QQeI7aS7li1V2FwJdzrDKoXADk8BlzhD02HHGAvR/AHx/Z/wBnQeC72Sf7d5s0lifKHl+XgOU3DndnzW+YYxxnoKoftI+IbhtR0nw0q7bVIvt8hyD5jkvGvbI2hX787+nAoAv/AAq+Mfh3SvBdvo3iS+ntLqwzHHNIkk4mjJJXG1SV2jC7TxgLg9QvnnxD1yb4o/E1U8PWsl0mxLKxVYyrzKpZi7ZPA3M5yduFAJAwa7jwZoXwP1PS3aa7k88PuZdcvjbSoDkBRsZEYfKT8u4jdyRwB6npmu/DjRPN/snVfClh52PM+yXFvFvxnGdpGcZPX1NAB8L/AAxceEfh/p2l30EEN+N8t0IcHLsxI3MPvMF2qTz93AJAFZ/xt/5JDrv/AG7/APpRHXoFeN/HnxtpFt4TvPCkVxHPq108PmwKTmCMMJNzHBGfkUbSQcOG6dQDzD4DalNY/FSzt4ljKX9vNbylgchQhlyvPXdGo5zwT9R9Z18YfC3W9O8OfEfSdW1a4+z2MHneZLsZ9u6F1HCgk8kDgV9V63498M6D4f8A7audXtJbR0drf7PMsjXJU7SsQB+chiAccDPJAyaAPiSvv+vgCvuPQPEtj4x8NLqugXcZEqFVM0e4wS4+7IgYHIJGQGGRgg4INAHw5X2/4E/5J54a/wCwVa/+ilr4gr3+w/aRS38PWyXmhT3erx7I5m+0LHHKAg3S5C/KxbPyBcAH73agD1PwJ/zMv/Yeuv8A2WutrhvhdqUOs6Nq+qW6yLBe6tNcRrIAGCuqMAcEjOD6mu5rKj8CO7Mv96l8vyQUUUVqcIUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAVyXwz/AOSe6X/21/8ARr11tebeD/EVx4f8LWWl3fhjxG88G/c0VgSp3OzDGSD0I7VjOSjUTfZ/oenhaM62EqQpq75oP5Wn/mek0VyX/Cd/9Sp4o/8ABd/9lR/wnf8A1Knij/wXf/ZU/bQ7mX9m4r+X8V/mdbRXJf8ACd/9Sp4o/wDBd/8AZUf8J3/1Knij/wAF3/2VHtodw/s3Ffy/iv8AM62iuS/4Tv8A6lTxR/4Lv/sqP+E7/wCpU8Uf+C7/AOyo9tDuH9m4r+X8V/mLr/xL8H+F9UbTNY1qOC8VA7RLDJKUB6btikA45wecEHoRXhni7W5fjh4+0TS/D8N9HYQxYk8+BM2+5/3sx2tyoUR8FhkjA5bnG8S/D3xVrPirV9Ut9Ev1gvb2a4jWS0lDBXcsAcKRnB9TXqfwytbf4faHJEfDfii41O82vezLp52ErnaiAt91dzckZOSTjhQe2h3D+zcV/L+K/wAz12wsbfTNOtrCzj8u1tYkhhTcTtRQAoyeTgAdasVyX/Cd/wDUqeKP/Bd/9lR/wnf/AFKnij/wXf8A2VHtodw/s3Ffy/iv8w8Cf8zL/wBh66/9lrra5LwClx9j1q4uLK6tPtWrT3Ecd1EY32MFIJB/L8DXW0UfgQZl/vU/l+SCvGv2gPtn9lab9h8/zvKut3kZ3eXiPzM4/h2bt3bbnPGa9lrhfHCadN4p8NQas8SafPFfw3Bll8tSjwhSC2RjOcdR1orfD81+aHlqbrtL+Wf/AKRI8y/ZovrePUfEVg0mLqaKCaNNp+ZELhjnpwZE/P2NfQ9fIN54J8QeC9cGp+H9c0u8+xbp4Lyy1CEPgZ4MTNuLFRygDA7tuW5Fbmm/HD4i2Nu0VxZ22oOXLCW6sWVgMD5R5ZQY4z0zyeemL549zn+q1/5H9zPpq/vrfTNOub+8k8u1tYnmmfaTtRQSxwOTgA9K+WPgDpn2/wCKEFz53l/2faTXO3bnzMgRbc54/wBbnPP3cd8izd6t44+KfnRarqNlpmmv+7SGS+Syt43GxsyRndLKp28ZBw7ZBABFe2eDLDwH4F0t7HR9Z08mV981xPexNLMecbiMDABwAAAOT1JJOePcPqtf+R/czuqKyf8AhKfD3/Qe0v8A8DI/8aP+Ep8Pf9B7S/8AwMj/AMaOePcPqtf+R/czWorJ/wCEp8Pf9B7S/wDwMj/xo/4Snw9/0HtL/wDAyP8Axo549w+q1/5H9zNaisn/AISnw9/0HtL/APAyP/Gj/hKfD3/Qe0v/AMDI/wDGjnj3D6rX/kf3M1qKyf8AhKfD3/Qe0v8A8DI/8aP+Ep8Pf9B7S/8AwMj/AMaOePcPqtf+R/czWorJ/wCEp8Pf9B7S/wDwMj/xo/4Snw9/0HtL/wDAyP8Axo549w+q1/5H9zNaisn/AISnw9/0HtL/APAyP/Gj/hKfD3/Qe0v/AMDI/wDGjnj3D6rX/kf3M1qKyf8AhKfD3/Qe0v8A8DI/8aP+Ep8Pf9B7S/8AwMj/AMaOePcPqtf+R/czWrxP9o3X77T/AA/pei27Rraao8jXWVyzCIxsqg9hubJ7/KOcZB9W/wCEp8Pf9B7S/wDwMj/xrh/ibonhX4g6HHEPEml2+p2e57KZr1NgLY3I4B+621eQMjAIzypOePcPqtf+R/czrvBOjaFofhLT7fw4n/EtliW4jmYEPPvUHzHyASxGOoGOAAAABYvvCfhvU7yS8v8Aw/pV3dSY3zT2UcjtgADLEZOAAPwr5l8GfEbxj8PUfR0sF1DT47jLQShnEeGO8QyIcAN1z8y5+YDk59M/4aDtf7H83/hEtX/tP/n3yPJ+9/z1xu+7z9zrx70c8e4fVa/8j+5l74nfDnwFb+F73XrrTP7Oaxi3A6ZiDzTkhIyoRlG53UbtmRxk4FZf7N8Grw+GtXe5ikTSZrhHsmZQA8mCspXuR8sYz0yCByGrjdT1fxH8Z9RistTuNL8OaRYymUG5maPcHIA+Vm/eyKobBAUctkruFe6eHLrwf4W8P2ei6brmni0tUKp5l9GzMSSzMTnqWJPGBzwAOKOePcPqtf8Akf3M8K8bfETVPizrlh4U8Nwz2umXMqJ5co+ed+peXZuxGnJwM/dLHOAF9u+F0EMPgCweKKNHmeV5WVQC7eYy5b1O1VGT2AHar8ep+DIdUm1SK+0BNQmTZLdrLCJXXjhnzkj5V4J7D0qp8M/+Se6X/wBtf/Rr1nzJ1VZ9H+h1qlOnganPFr34brymdbRRRWx5pDdWsN7Zz2lwm+CeNo5FyRuVhgjI56GuY/4Vn4Q/6BH/AJMy/wDxddbRUSpwl8SudFHF4igmqNRxv2bX5HJf8Kz8If8AQI/8mZf/AIuj/hWfhD/oEf8AkzL/APF11tFT7Cl/KvuNv7Ux3/P6f/gT/wAzkv8AhWfhD/oEf+TMv/xdH/Cs/CH/AECP/JmX/wCLrraKPYUv5V9wf2pjv+f0/wDwJ/5nJf8ACs/CH/QI/wDJmX/4uj/hWfhD/oEf+TMv/wAXXW0Uewpfyr7g/tTHf8/p/wDgT/zOS/4Vn4Q/6BH/AJMy/wDxdH/Cs/CH/QI/8mZf/i662ij2FL+VfcH9qY7/AJ/T/wDAn/mcl/wrPwh/0CP/ACZl/wDi6P8AhWfhD/oEf+TMv/xddbRR7Cl/KvuD+1Md/wA/p/8AgT/zOS/4Vn4Q/wCgR/5My/8AxdH/AArPwh/0CP8AyZl/+LrraKPYUv5V9wf2pjv+f0//AAJ/5nJf8Kz8If8AQI/8mZf/AIuj/hWfhD/oEf8AkzL/APF11tFHsKX8q+4P7Ux3/P6f/gT/AMzkv+FZ+EP+gR/5My//ABdH/Cs/CH/QI/8AJmX/AOLrraKPYUv5V9wf2pjv+f0//An/AJnJf8Kz8If9Aj/yZl/+Lrc0bQtN8P2b2ml232eB5DIy72fLEAZyxJ6AVo0VUaUIu8UkZ1cbiq0eSrUlJdm21+IUUUVZylTU9Ms9Z06Wwv4fOtZcb03Fc4II5BB6gVzv/Cs/CH/QI/8AJmX/AOLrraKiVOEneSTOmjjMTQjy0qkorybX5HJf8Kz8If8AQI/8mZf/AIuj/hWfhD/oEf8AkzL/APF11tFT7Cl/KvuNf7Ux3/P6f/gT/wAzkv8AhWfhD/oEf+TMv/xdH/Cs/CH/AECP/JmX/wCLrraKPYUv5V9wf2pjv+f0/wDwJ/5nJf8ACs/CH/QI/wDJmX/4uj/hWfhD/oEf+TMv/wAXXW0Uewpfyr7g/tTHf8/p/wDgT/zOS/4Vn4Q/6BH/AJMy/wDxdH/Cs/CH/QI/8mZf/i662ij2FL+VfcH9qY7/AJ/T/wDAn/mcl/wrPwh/0CP/ACZl/wDi6P8AhWfhD/oEf+TMv/xddbRR7Cl/KvuD+1Md/wA/p/8AgT/zOS/4Vn4Q/wCgR/5My/8AxdH/AArPwh/0CP8AyZl/+LrraKPYUv5V9wf2pjv+f0//AAJ/5nJf8Kz8If8AQI/8mZf/AIuj/hWfhD/oEf8AkzL/APF11tFHsKX8q+4P7Ux3/P6f/gT/AMyG1tYbKzgtLdNkEEaxxrknaqjAGTz0FTUUVqcTbbuwrkvEH/JQvB3/AG+/+ihXW1yXiD/koXg7/t9/9FCsq3w/Nfmjty7+M/8ABU/9IkdbWf8A2Fo/9sf2v/ZVj/af/P79nTzvu7fv43fd469OK0KK1OEKrzWFncXlteTWkEl1a7vs8zxgvFuGG2seVyODjrViigAooooA5vX/AAB4U8UO0usaHaTzs4dp1Bilchdo3SIQxGOME44HoKseGvB+geD7eeDQdNjs0ncPKQ7OzkDAyzEnA5wM4GT6mtyigDHvvCfhvU7yS8v/AA/pV3dSY3zT2UcjtgADLEZOAAPwrYoooAr31hZ6nZyWd/aQXdrJjfDPGJEbBBGVPBwQD+FU9N8NaDo1w1xpeiabYzshRpLW1SJiuQcEqAcZAOPYVqUUAFcnr/w08H+KNUbU9Y0WOe8ZAjSrNJEXA6btjAE44yecADoBXWUUAcPY/B/wDp95HdQ+HIHkTOBPLJMhyCOUdip69xx1613FFFAHBz/Bj4fXNxLO/h2MPI5dhHczIoJOeFVwFHsAAO1al98OPBWoWclrN4X0pI3xkwWywuMEHh0AYdOx56dK6iigCnpWlWOh6Xb6ZpltHbWdumyKJOij+ZJOSSeSSSck1cqvf31vpmnXN/eSeXa2sTzTPtJ2ooJY4HJwAelfPmt/tIX06ahBo2ix2wdNlnczy73jO45kZMbSSu3C5wpByXBxQBT+O3jG48ReKIfB+lf6Ra2UqB0hQO812QRtUgknaG27QAdxcEHAx7v4H8N/8Ij4L0vQzJ5klrF+9cNkGRiXfacD5dzNjIzjGea+fPgF4RbWPGJ1y7spH0/TELRSsqmM3PG1eRyVUl+OVIQ5GRn6joA5Lx3/AMy1/wBh61/9mrra5Lx3/wAy1/2HrX/2autrKHxy+R3V/wDdaP8A29+YUUUVqcJ5NrPw80Xx/wCOvEsWrPdxPZvaNDLayBWAaH5lO4EEHap6Z+UYI5zB/wAM4+D/APoJa5/3/h/+NV2fh/8A5KF4x/7cv/RRrrayo/D83+bO7Mf4y/wU/wD0iJ5fa/AHwNb6dPbS299dTSbtt3NdESRZGBtCBU4PI3KeTzkcVJpvwG8B2Nu0VxY3eoOXLCW6u3VgMD5R5ewY4z0zyeemPTKK1OE5fw98OvCPhW8N5o+iQQXR6TOzyunBHys5JXIYg7cZ75rwj4m6hcfFX4ix6P4T06DUP7OiaNbu3IzOOC5aRsKI1bKrzgkkgneBX0/Xyp8E/iBpHgnVNTt9a8yG01FIz9qVS4iaPfgMqgkht5GRnBA4wSQAGpfs++N7G3WW3Gm6g5cKYrW5KsBg/MfMVBjjHXPI464y/wDhSXxD/wChe/8AJ23/APjlfWelarY65pdvqemXMdzZ3Cb4pU6MP5gg5BB5BBBwRUes65pfh7Tnv9Xv4LK1XI3zPjcQCdqjqzYBwoyTjgUAfHngzxnrXw48SvPBHIAH8m/0+fKCUKSCrAjKupzg4ypzwQSD73ffBTwV4wvJPEsN7qsceq4vQIJVVG8wBtwDxlhuzuwTxnoBwPCNburz4n/FCeXTrfbNqt2sVuhQjZGAEVnA3YwihnIyBhj0r7PoA8T1L9mzQZbdV0vXNStp94LPdKk6lcHgKoQg5xznseOeOo1f4MeGdY8P6Jorz6lb2mjpKsHkzLucyEM7OWU5JZc8YAycDGAPRKKAPH/+GcfB/wD0Etc/7/w//Gq7jwR4C0fwDp1zZ6SZ5ftMvmyTXJRpDgABdyqvyjkgHoWb1rqKKAPN/HXwZ8P+MpTe25/snUzktcW0SlJSW3FpE43Ny3zAg/NyTgCuLg/ZmhW4ia48VySQBwZEjsAjMueQGMhAOO+Dj0Ne+UUAcN8LtNh0bRtX0u3aRoLLVpreNpCCxVFRQTgAZwPQV3Ncl4E/5mX/ALD11/7LXW1lR+BHdmX+9S+X5IKKKK1OEKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAOS1nWfEP8Awl6aHoaaX/x4C8Z74Sf89ChAKH6dvXmsODxprN1cRW9v4l+H008rhI44792Z2JwAAGySTxiuc+NvTXf+wDb/APpfHXhHhTwlq/jPWRpejQxyThPMkaSQIsce5VLnPJALDgAn0BrCMXJttvc9WtWjQjTjGnF3inqtbn0t/wAJ3qf/AENfw7/8GLf/ABVdB/xcP/qV/wDyYrwXUvgD4k0fQ9U1W91PSvLsbR7kJC8jGTZ8zLygx8oYg88gDgHI8z0nTZtZ1mx0u3aNZ724jt42kJChnYKCcAnGT6Gq9l5sw+vf9O4fcfX11f8Ajex8j7Zd+ELfz5Vgh86SZPMkb7qLnqxwcAcmrH/Fw/8AqV//ACYrx3U/2a9Yi8r+yfEFjdZz5n2uF7fb0xjbvz364xgdc8ef6X4Y1nw58TtA0jV4J9Nvm1C1KsuxioaRcOh+ZGwc+oyCCOCKPZebD69/07h9x6Xf/tCanY6jc2f2XTLjyJXi863jZ45NpI3I3mcqcZB7itDwt8a9f8XeI7TQ7Cy0yO6ut+x54pAg2oznJEhPRT2qP48/D3S7fQ5fFmlWHk3wu1OoPHJtR0f5d5Qn72/Z93Gd7Eg9RxHwB0z7f8UILnzvL/s+0mudu3PmZAi25zx/rc55+7jvkHsvNh9e/wCncPuPoL/i4f8A1K//AJMUf8XD/wCpX/8AJiutoo9l5sPr3/TuH3HJf8XD/wCpX/8AJij/AIuH/wBSv/5MV1tFHsvNh9e/6dw+45L/AIuH/wBSv/5MUf8AFw/+pX/8mK62ij2Xmw+vf9O4fcclo2s+If8AhL30PXE0v/jwN4r2Ik/56BACXP17enNdbXJf81e/7gP/ALXrraKV7NN9Qx6jzQlGKV4p6BRRRWpwnMaz4n1Ky8Rpoul6F/aU5tBdsftaw7V3lf4hjqB371D/AMJB4v8A+hH/APKtF/hR/wA1e/7gP/teutrCKlJt8zWvl/kepVnRoRhH2MZXim23O936SS/A5L/hIPF//Qj/APlWi/wo/wCEg8X/APQj/wDlWi/wrraKr2cv5n+H+Rj9bo/8+IffP/5M5L/hIPF//Qj/APlWi/wo/wCEg8X/APQj/wDlWi/wrraKPZy/mf4f5B9bo/8APiH3z/8Akzkv+Eg8X/8AQj/+VaL/AAo/4SDxf/0I/wD5Vov8K62ij2cv5n+H+QfW6P8Az4h98/8A5M5L/hIPF/8A0I//AJVov8KP+Eg8X/8AQj/+VaL/AArraKPZy/mf4f5B9bo/8+IffP8A+TOS/wCEg8X/APQj/wDlWi/wo/4SDxf/ANCP/wCVaL/Cutoo9nL+Z/h/kH1uj/z4h98//kzkv+Eg8X/9CP8A+VaL/Cj/AISDxf8A9CP/AOVaL/Cutoo9nL+Z/h/kH1uj/wA+IffP/wCTPLvEvxeuPCFxBb674djtZ50LxxjUVkbaDjJCISBnIBOM4OOhrE/4aN0v/oEf+TLf/Gq7T4ifDLS/iDZxGWT7FqcGBDfJHvITOSjLkbl5JHIweQeWB+OKPZy/mf4f5B9bo/8APiH3z/8Akz7S/wCEg8X/APQj/wDlWi/wo/4SDxf/ANCP/wCVaL/Cutoo9nL+Z/h/kH1uj/z4h98//kznfDniO81nUdTsL/Sf7OurDyt6faRNnzASOQAOgHr1rXvtK07U/L+32Frd+XnZ58Kybc4zjI4zgflXO+H/APkoXjH/ALcv/RRrraKfvR97XV/mGNSpV06K5fdi9G9LxTdrtvdvqZP/AAi3h7/oA6X/AOAcf+FH/CLeHv8AoA6X/wCAcf8AhWtRV8kexh9ar/zv72ZP/CLeHv8AoA6X/wCAcf8AhR/wi3h7/oA6X/4Bx/4VrUUckewfWq/87+9mT/wi3h7/AKAOl/8AgHH/AIUf8It4e/6AOl/+Acf+Fa1FHJHsH1qv/O/vZk/8It4e/wCgDpf/AIBx/wCFH/CLeHv+gDpf/gHH/hWtRRyR7B9ar/zv72ZP/CLeHv8AoA6X/wCAcf8AhR/wi3h7/oA6X/4Bx/4VrUUckewfWq/87+9mT/wi3h7/AKAOl/8AgHH/AIUf8It4e/6AOl/+Acf+Fa1FHJHsH1qv/O/vZk/8It4e/wCgDpf/AIBx/wCFH/CLeHv+gDpf/gHH/hWtRRyR7B9ar/zv72ZP/CLeHv8AoA6X/wCAcf8AhR/wi3h7/oA6X/4Bx/4VrUUckewfWq/87+9mT/wi3h7/AKAOl/8AgHH/AIUf8It4e/6AOl/+Acf+Fa1FHJHsH1qv/O/vZk/8It4e/wCgDpf/AIBx/wCFH/CLeHv+gDpf/gHH/hXGeHfB+g+ILzxHd6pY/aJ01q5jVvOdMKCDjCsB1Jrc/wCFZ+EP+gR/5My//F1hG8ldRX3/APAPTrezoz9nOvO67R0/9LRr/wDCLeHv+gDpf/gHH/hR/wAIt4e/6AOl/wDgHH/hWR/wrPwh/wBAj/yZl/8Ai6P+FZ+EP+gR/wCTMv8A8XVcsv5V9/8AwDL21D/n/U/8BX/yw1/+EW8Pf9AHS/8AwDj/AMKP+EW8Pf8AQB0v/wAA4/8ACuE8b/C6Obw+IvB2nWkGpvcRq0tzcSnZET8zLkkZHBOQfl3YBbbXzhqt34i0PVLjTNT8y2vLd9ksTxplT+WCCMEEcEEEZBo5Zfyr7/8AgB7ah/z/AKn/AICv/lh9lf8ACLeHv+gDpf8A4Bx/4VoWtrb2VulvaW8UECZ2xxIEVcnJwBx1JNcB4T8AeGNT8G6Hf3mmeZdXWn280z+fKNztGpY4DYGST0o8Q+HPhn4Usxda5HBZRt9wPdTF5MEA7UDFmxuGcA4zk8U0praK+/8A4BM5YWatOtN+sV/8sPRaK4qx8A+BNTs47yws4Lu1kzsmgvZJEbBIOGD4OCCPwqx/wrPwh/0CP/JmX/4uner2X3/8Ay9ngf8An5P/AMAX/wAsOtorkv8AhWfhD/oEf+TMv/xdH/Cs/CH/AECP/JmX/wCLovV7L7/+AHs8D/z8n/4Av/lh1tFcl/wrPwh/0CP/ACZl/wDi6P8AhWfhD/oEf+TMv/xdF6vZff8A8APZ4H/n5P8A8AX/AMsOtorkv+FZ+EP+gR/5My//ABdH/Cs/CH/QI/8AJmX/AOLovV7L7/8AgB7PA/8APyf/AIAv/lh1tFcl/wAKz8If9Aj/AMmZf/i6P+FZ+EP+gR/5My//ABdF6vZff/wA9ngf+fk//AF/8sOtorkv+FZ+EP8AoEf+TMv/AMXR/wAKz8If9Aj/AMmZf/i6L1ey+/8A4AezwP8Az8n/AOAL/wCWHW0VyX/Cs/CH/QI/8mZf/i6P+FZ+EP8AoEf+TMv/AMXRer2X3/8AAD2eB/5+T/8AAF/8sOtorkv+FZ+EP+gR/wCTMv8A8XR/wrPwh/0CP/JmX/4ui9Xsvv8A+AHs8D/z8n/4Av8A5YdbRXJf8Kz8If8AQI/8mZf/AIuj/hWfhD/oEf8AkzL/APF0Xq9l9/8AwA9ngf8An5P/AMAX/wAsOtorkv8AhWfhD/oEf+TMv/xdH/Cs/CH/AECP/JmX/wCLovV7L7/+AHs8D/z8n/4Av/lh1tFcl/wrPwh/0CP/ACZl/wDi6P8AhWfhD/oEf+TMv/xdF6vZff8A8APZ4H/n5P8A8AX/AMsOtorkv+FZ+EP+gR/5My//ABdVPBmmWejeMvFlhYQ+TaxfY9ibi2MxsTyST1Jo55qSUktfPy9Cvq2GnSnOlOTcVezil1S3Un37HcUUUVqeeFFFcl4+e4+x6Lb297dWn2rVoLeSS1lMb7GDAgEfn+AqZy5Y3N8NQ9vVVO9rnW0VyX/CCf8AU1+KP/Bj/wDY0f8ACCf9TX4o/wDBj/8AY1HPP+X8Tf2GF/5/f+Ss62ivB/i/fal8PrfSV0vXtdnnvnlLPdagzKioF4CqASSXHOeMHg548r/4Wp4w/wCgzd/+BU3/AMXRzz/l/EPYYX/n9/5Kz7Morz7w14Wm1nwrpGqXHijxIs97ZQ3Eix6gQoZ0DEDIJxk+prU/4QT/AKmvxR/4Mf8A7Gjnn/L+Iewwv/P7/wAlZ1tFcl/wgn/U1+KP/Bj/APY0f8IJ/wBTX4o/8GP/ANjRzz/l/EPYYX/n9/5KzraK5L/hBP8Aqa/FH/gx/wDsaP8AhBP+pr8Uf+DH/wCxo55/y/iHsML/AM/v/JWdbRXJf8IJ/wBTX4o/8GP/ANjR/wAIJ/1Nfij/AMGP/wBjRzz/AJfxD2GF/wCf3/krOtorkv8AhBP+pr8Uf+DH/wCxo/4QT/qa/FH/AIMf/saOef8AL+Iewwv/AD+/8lZ1tcx4n0bWb3WdG1TRXsBPp/n5W9L7W8xQv8Iz0B7jtUP/AAgn/U1+KP8AwY//AGNH/CCf9TX4o/8ABj/9jUz55Kzj+Jth1h6FT2kavdaxdrNNP8GH/Fw/+pX/APJij/i4f/Ur/wDkxR/wgn/U1+KP/Bj/APY0f8IJ/wBTX4o/8GP/ANjUcs/P7zo9the8P/AH/mH/ABcP/qV//Jij/i4f/Ur/APkxR/wgn/U1+KP/AAY//Y0f8IJ/1Nfij/wY/wD2NHLPz+8PbYXvD/wB/wCYf8XD/wCpX/8AJij/AIuH/wBSv/5MUf8ACCf9TX4o/wDBj/8AY0f8IJ/1Nfij/wAGP/2NHLPz+8PbYXvD/wAAf+Yf8XD/AOpX/wDJij/i4f8A1K//AJMVwfxM1GH4dW+nM2r+K9QnvnkCINWESqqBdxLbGOcuuBj15GOfLLr4taw+owPZ3WtRWI2+dDNqzySPz821wqhcjAGVbB556Ucs/P7w9the8P8AwB/5n0f/AMXD/wCpX/8AJij/AIuH/wBSv/5MVz/gnS/+Ex8IWOv/ANv+KLP7V5n7j+1fM27ZGT72wZztz0710H/CCf8AU1+KP/Bj/wDY0cs/P7w9the8P/AH/mH/ABcP/qV//Jij/i4f/Ur/APkxWTrWi3Hhy80G4t/EWvXHn6tb28kd1el0ZGJJBAAz0x+deh1UIuTabat5kYirGlGM4RhJSv8AZtt8zkv+Lh/9Sv8A+TFH/Fw/+pX/APJiutoq/ZebOT69/wBO4fccl/xcP/qV/wDyYo/4uH/1K/8A5MV1tFHsvNh9e/6dw+45L/i4f/Ur/wDkxR/xcP8A6lf/AMmK62ij2Xmw+vf9O4fcefeI9D8e+JfD95o1xdaFbQXaBJJLSSdJNuQSASCMEDaRjkEjvXmn/DOOp/8AQStP+/7f/Gq+jKKPZebD69/07h9x594c0Dxp4W8P2ei6avhsWlqhVPMa4ZmJJZmJ9SxJ4wOeABxWp/xcP/qV/wDyYrraKPZebD69/wBO4fccPeaN4y1m80r+1H0FbWzv4rxvspmDnYeg3AjoT6dua7isPxhrNx4f8LXuqWiRPPBs2rKCVO51U5wQehPes7/i4f8A1K//AJMVKapyaV2zaUKmKpRnJxhFNpdNdG/zR1tFeReN/iV4j8BfZo9Vfw9NdXHKWtoszyBOfnILABcjHJ5OcA4OOQ/4aO1P/oG2n/fhv/jtV7XyZj9R/wCnkPvPVHTxHo3jLXb+w8O/2ja3/wBn2P8AbY4ceXHg8HJ6k+nSrf8AwkHi/wD6Ef8A8q0X+FeQf8NHan/0DbT/AL8N/wDHa9f/AOLh/wDUr/8AkxWSutub7l/kehLknb2ipNpJX5p9EktpJbLsH/CQeL/+hH/8q0X+FH/CQeL/APoR/wDyrRf4Uf8AFw/+pX/8mKP+Lh/9Sv8A+TFF5d5fcv8AInko/wAlL/wKf/yQf8JB4v8A+hH/APKtF/hXlHjr4V634u1w6vYeF/7IuJsm6VL2GVJn/vgZXax5z1z1wDkn1f8A4uH/ANSv/wCTFH/Fw/8AqV//ACYovLvL7l/kHJR/kpf+BT/+SPAP+FC+L/8An2/8iRf/AByj/hQvi/8A59v/ACJF/wDHK9//AOLh/wDUr/8AkxR/xcP/AKlf/wAmKLy7y+5f5ByUf5KX/gU//kjk/A3h3VfAulxwWPgGOa/KEXGoPqUAlmJwSM4yqZUYQHAwM5OSes/4SDxf/wBCP/5Vov8ACj/i4f8A1K//AJMUf8XD/wCpX/8AJii8u8vuX+QclH+Sl/4FP/5IP+Eg8X/9CP8A+VaL/Cj/AISDxf8A9CP/AOVaL/Cj/i4f/Ur/APkxR/xcP/qV/wDyYovLvL7l/kHJR/kpf+BT/wDkg/4SDxf/ANCP/wCVaL/Cj/hIPF//AEI//lWi/wAKP+Lh/wDUr/8AkxR/xcP/AKlf/wAmKLy7y+5f5ByUf5KX/gU//kg/4SDxf/0I/wD5Vov8KP8AhIPF/wD0I/8A5Vov8KP+Lh/9Sv8A+TFH/Fw/+pX/APJii8u8vuX+QclH+Sl/4FP/AOSJvBNhqVlZ6rJqll9jnvNTmu1h81ZNquF/iU46gjt0rp642DWfFVl4p0jS9aTRjBqHnYayEu5fLTd/EcdSOx712Va0muWy6dzhzCM/a+0nb3ldcu1rtfoFFFFanCFFcl4ru9Y/4SHQNK0rVP7P+3/aPMl+zpL9xFYcN+I6jrR/wj/i/wD6Hj/ykxf41k6ju0ot29P8zujgo+zjOdWMeZXSfNe12ukWt0+p1tFcl/wj/i//AKHj/wApMX+NH/CP+L/+h4/8pMX+NHtJfyv8P8w+qUf+f8Pun/8AIHW0VyX/AAj/AIv/AOh4/wDKTF/jR/wj/i//AKHj/wApMX+NHtJfyv8AD/MPqlH/AJ/w+6f/AMgdbRXJf8I/4v8A+h4/8pMX+NH/AAj/AIv/AOh4/wDKTF/jR7SX8r/D/MPqlH/n/D7p/wDyB1tFcl/wj/i//oeP/KTF/jR/wj/i/wD6Hj/ykxf40e0l/K/w/wAw+qUf+f8AD7p//IHW0VyX/CP+L/8AoeP/ACkxf415Z4k+LVz4f1TUtLTxXqV5eWTvDlNGt0iaVeCu5n3ABvlJ2noSARjJ7SX8r/D/ADD6pR/5/wAPun/8gfQFFfLdj8d/Ecl5Gt/qE8Fqc75ILWCV14OMKVUHnH8Q9eele7/8I/4v/wCh4/8AKTF/jR7SX8r/AA/zD6pR/wCf8Pun/wDIHW0VyX/CP+L/APoeP/KTF/jVRH8R6N4y0Kwv/EX9o2t/9o3p9ijhx5ceRyMnqR6dKHVa3i/w/wAyo4GE7+zrRbSbt7/RNveKWy7ncUUUVqeeFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAHh/xn8Jav4x8UQ2Wiwxz3cGnR3HktIELqJXQhS2Bn5weSOAe+Aeo+C/gvUfBnhK5i1m0gt9Su7tpWCMruIwoVFZl4OCHYAEgb+xJFbf/NXv+4D/AO1662sqX2vU7sd/y7/wI5/x3/yTzxL/ANgq6/8ARTV8geBP+Sh+Gv8AsK2v/o1a+r/ipqf9kfC/xDc+T5u+0Ntt3bcecRFuzg9N+cd8Y4618oeBP+Sh+Gv+wra/+jVrU4T7frg/Gnwzh8W+KtD8QQ6pJpt5pjqWkjhEjSqrh0A3HapU7+SrZ3cggYrvKKAPN/jrY2938KNSmnj3yWksE0B3EbHMixk8dfldhz6+uK88/Zp02GXWdf1Rmk8+3t4rdFBG0rIzMxPGc5iXHPc9e3ofx1vre0+FGpQzybJLuWCGAbSd7iRZCOOnyox59PXFfOnw80TxN4g8StY+FdQk0+8+zu8twt00AWIFc7inzEFtgwAeSD0BIAPrPU/HPhbRdYi0nUtesbW+kz+6klA8vChv3h6R5BBG4jdnjNdBXgd7+zTD/ZcYsPEkh1BEkLme2Aimb+AAA5jHYn5/UAdK5z9nzxPcaf40bw/JPO1jqUUhjgGCizoN285+78iODjr8uRwCAD6foryf46eBdU8XaHY3+kDz7jSvNZrRVy8yPsyU9WGwfL3yccgA/LFAH3/WPfeLPDemXklnf+INKtLqPG+Ge9jjdcgEZUnIyCD+NXNJ02HRtGsdLt2kaCyt47eNpCCxVFCgnAAzgegr4w8d+FF8FeLLnQl1KO/MCIzSrE0ZUsobaynIzgg8EjBHOcgAH1FpurabrPxVa40vULS+gXRCjSWsyyqG88HBKkjOCDj3FdzXhfwR8Map4X8QGHV4Ps9xeaY12sDffjRpEUBx/C3yE47ZGcHIHulZUvtep3Y7/l3/AIEFFFFanCcVfX9npnxSkvL+7gtLWPQRvmnkEaLm4AGWPAySB+NdTpurabrNu1xpeoWl9ArlGktZllUNgHBKkjOCDj3FeRfFzwZfeN/FBsNMkjF5baSt3FE/AmKysuzdnCkhzgnjIAOAcj5prKl9r1O7Hf8ALv8AwI+/6K4f4daD9n+D2j6V9tnT7Xp5k+0W58qSLz90nyHnDL5mA3qucdq+cPiJ4O1j4feI4oJtTnvI7i0C298FeMvHs8pojknGF+UqGI2MvQHFanCfY9FeF/s9eENS0+1l8VS6hGNP1K3aKKzjZiWZZSN7jgArsYLjPDnkdDc+POjeJBpkviHTfEE9tpEFotte6elxJGJN0m3cFX5X3eZghscL3zgAHtFFeB/s5ab4jhTUL+VpI/Dc6ERRyHiW4DAb0GOgUMrEYBOByV+WT4zfFu4028HhzwrqflXEe9dRuYVBKEjAiR+zDLFiBlSFAYEMKAPeKK+XNK+E/wASvFiW+v32pSWl5E+23bV7qdbpAjZDD5WZAGJIyQc8gYIJk0zx341+FXjmXTvF1xfanaNgTRz3DTb48nbNAzn68cA8q2GGVAPp+iqcGq2Nzo0WsJcxjT5LcXS3EnyKIiu7ed2No2884x3rwS5+LHjv4geIWsPh9Z/Yo7aJ5WRzA0kyBwA7GUbV6r8q5wSeWGMAH0PRXzJqui/GjwU9xqravqV5aWKb3uUv/tMRUryfKkOSFyckpxgt0Ga9f+FXxAh8d+GgZfMGrWCRxXwZRh2IOJFIAGG2scDG0gjGMEgHYatpsOs6NfaXcNIsF7byW8jRkBgrqVJGQRnB9DXwxBDfa7rMUCGS61C/uAimST5pZXbHLMepY9Se/NfedfGFg1nffGS2bSJfs9jP4gQ2clvGE8uNrgbCisuBgEYBXA6EdqAPs+iuP+KGvXHhv4f6jqFjqkGnX67BayzKG3vuBKKpVtzMoYDjjqSACR8waZrvxH1vzf7J1XxXf+TjzPslxcS7M5xnaTjOD19DQB9ReH/+SheMf+3L/wBFGutrx74DaVfaLZatYalbSWt2iW7vDJwyh/MdcjsdrDg8jocHivYayo/D83+bO7Mf4y/wU/8A0iIUUUVqcIUVwOs6FpviD4pJaapbfaIE0USKu9kwwnIzlSD0JrzHxD4y+Gdhrg0nStD+1RiXyZ9Ue6mMEPQeYqqxaZVyxIBXO35SQQaxU5tvlS08/wDgHoyw2Fpxi6tSSbSekU1r5ua/I+jKK+P7m38Yadql1qT+FNSk0uF5ZfKu9MniiWLnBba25Qo5/wBYcY5Zuc+ifCq+8IePZZNKv/DP2fV4YnuHeC4l8h4wyqMZkLBvnAxyOM55wHer2X3/APAI9ngf+fk//AF/8sPfKK+V/i9pmq+FfFko07S5LDQSka21wqtIkrFctudi2H3Bxt44UHHOTzfgaXVPEfjnRdJZvtEM92nnxYVN0KndJzxj5Ax4OfTnFF6vZff/AMAPZ4H/AJ+T/wDAF/8ALD7MorhtW+G/h6HRr6XS9Cjn1BLeRrWKS6lCvKFOxT+8HBbA6j6ivmTU/F18NRijfRLHTpLSUi4tVWceYQRlJBJIzLjBHylTyec4wXq9l9//AAA9ngf+fk//AABf/LD7Uory7wh4P0LUPBFrrPifw9HpV5sle6jkluIFhVXYAkSPlRtUNknvnpXknjLx54Q/e2XhDw56p/aF3PL/ALQzHHv/AN1gX9wUovV7L7/+AHs8D/z8n/4Av/lh9WUV8VzXfifw1eW0+uaPOkb7tkGpWkkKTYGDyNjHG4Hg+meOD7Pokvw58X+ENSu9JsPs2sWunSXEtnJcSl4GCnkEth1DAcj1XcATik5VEr2X3/8AALp0MFOagqktdPgX/wAme20Vk+Fv+RQ0X/rwg/8ARa1rVrF3SZxVYeznKHZ2CiiimZnJeBP+Zl/7D11/7LXW1xXhO+t9M07xff3knl2trrN5NM+0naihSxwOTgA9K8d8SfH3xLreox2XhK0/s+N5QkJ8pbi5nJJCjaQVG7K/KATkcMQayo/AjuzL/epfL8kfS9FfPF944+M/gazkvPEOlwXdrJjE09vHIkGCBy1uwC7iyj5+uOO9cP4s+Mnizxdpy2FxNBYWvzCVNPDxeeGGNrksSVwT8vAOeQcDGpwn1/XyB8bf+Sva7/27/wDpPHX0f8L7jxBc/D/Tm8TRTpqS74ybkMJmRWIUyKyghsDHfIAYnLHHiH7R3/JQ9P8A+wVH/wCjZaAPf/An/JPPDX/YKtf/AEUted/Hvwfr/im30WfRNNkvUsEuXuBG67lBEZGFJBYna3Cgnj3FekQalpGjeBYtUsFkbRbLTBcQLGCWNukW5QN5BztA+8QfWvG/+Gmv+pR/8qX/ANqoA9A+DvhHUfBvgY2WrL5V9c3clzJBlW8rIVAu5WIbIQNkf3sdq9ArD0rxRY6h4Kt/FEskcNm9l9sn2P5ohAXdIuVGSVIYHAzlTxnivO/+GjvB/wD0Ddc/78Q//HaAPYKK8X1D9ozQoNDsLqx0ye71KfP2myeQxC2x1zJtIfJ6YHI5O08HE039pOZ/EDNqmhxxaKyEKlqxkuI2wOSzFVcEg8YXAI5OOQD6Dorg9f8Ai94Q0bw02rWurWmpSugNvZ206mWRmGQGXrGPUsOOmCcKfKJv2k9ebVBJBoempp+9SYHZ2l28bh5gIGTzg7OMjg45APpOiuX8C+OtL8eaGL+wPlXEeFurR2y9u57H1U4OG747EECPxx8Q9F8AW9nLqyXcr3jssMVrGGYhQNzHcQABuUdc/MMA84AOsory+w+Pngq+1i20/dfW6z7B9ruIlSGNmUHDndkYJ2k42g852/NWP4h/aM0LT7wQaHpk+rxj787yG2Q8AjaCpY9SDkLjHGQc0Ae0UV5H4M+Pei+I9Uew1i0j0IlN0M892HicjJKs5VQhx0zweRkHAMnjr46aX4R1w6RYaf8A2vcQ5F0yXPlJC/8AcB2tuYc56Y6ZJyAAesUV43oH7RXhzUHWLWtPu9JdnI8xT9oiVQuQWKgPknIwEPbnrj2SgAoqnquq2Oh6XcanqdzHbWdum+WV+ij+ZJOAAOSSAMk14/qX7SegxW6tpeh6lcz7wGS6ZIFC4PIZS5JzjjHc88cgHtlFeV6b+0F4Ivrhorg6lp6BCwlurYMpOR8o8tnOec9McHnpn0jVdVsdD0u41PU7mO2s7dN8sr9FH8yScAAckkAZJoAuUV4//wANHeD/APoG65/34h/+O1seK/jV4W8P6Hb32m3UGtXV1gw2ltOAQvBJkOCY8A9GG4njHDEAHpFFef8Aw7+LGl/ECWWyjs57HU4YjNJbufMQpu25VwBnGUzkL97jOCax/GPx68P+HLz7FpMH9uXC7TJJBcKsCggnAkAbcw+XgDHPXIIoA9YrkvD/APyULxj/ANuX/oo15Z/w01/1KP8A5Uv/ALVXZ/CnxP8A8JjqfiLX/sf2P7V9m/ceb5m3arp97Aznbnp3rKp8UfX9Gd2E/g1/8C/9Lgem0UUVqcIVyXjv/mWv+w9a/wDs1dbXHfEKeG1t/D1xcSxwwRa3bPJJIwVUUBySSeAAOc1lW+Bndlv+9R+f5MyPiV8W7f4e6jY2C6Z/aN1cRNNInnmHykzhTnYwbcQ/02+4rQ+GvxHt/iJp19Mtl9hurOVUkt/NMvyMMq+7ao5IcY7bfcV4J8cfEekeJvGtleaNfR3lummRI0iAgBizvjkDna65HY5BwQQOs+CnxE8K+EfBt5Ya5qv2S6k1B5lT7PLJlDHGAcopHVT+VanCe/31hZ6nZyWd/aQXdrJjfDPGJEbBBGVPBwQD+FfEHiyxt9M8Za5YWcfl2trqFxDCm4naiyMFGTycADrX23purabrNu1xpeoWl9ArlGktZllUNgHBKkjOCDj3FfFnjv8A5KH4l/7Ct1/6NagD7D8FwtbeBfD0DmMvHplsjGORXUkRKOGUkMPcEg9q3K87n+OPw+ht5ZU1qSd0QssUdnMGcgfdG5AMnpyQPUijQPjb4I151ibUJNMnZyqx6igiBAXO7eCUA6gZYHIxjkZAPRKKKx77xZ4b0y8ks7/xBpVpdR43wz3scbrkAjKk5GQQfxoA2KKr/b7P+zv7R+1wfYfK8/7T5g8vy8bt+7ptxznpiq+ma7o+t+b/AGTqtjf+TjzPslwkuzOcZ2k4zg9fQ0AaFFV76/s9Ms5Ly/u4LS1jxvmnkEaLkgDLHgZJA/GvJ/8Aho7wf/0Ddc/78Q//AB2gD2CivL9G+PngrV9RSzla+03fgLNfRKsZYkAAsjNt65y2FAByRXqFABRXP/8ACd+D/wDoa9D/APBjD/8AFVqaVqtjrml2+p6Zcx3NncJvilTow/mCDkEHkEEHBFAFyisvUvEug6NcLb6prem2M7IHWO6ukiYrkjIDEHGQRn2NR6D4q0LxP9s/sXU4L37HL5M/lE/K3Y89VODhhlTg4JwaANiiiuX8WfELw14K2x6zf+XdSRNLFaxRtJJIB7AYXJ4BYgEg88HAB1FFeb2Px18A3dnHPNqk9lI2cwT2khdMEjkorLz14J6+vFZ/iH9oHwjplmG0fz9Zum6RpG8CLgj7zOuRkE42q3TBxnNAHeeJfB+geMLeCDXtNjvEgcvES7IyEjBwykHB4yM4OB6CvjDxLpsOjeKtX0u3aRoLK9mt42kILFUcqCcADOB6Cvr/AMD/ABD0Xx/b3kukpdxPZuqzRXUYVgGB2sNpIIO1h1z8pyBxn5M8d/8AJQ/Ev/YVuv8A0a1AH1v8OLG30/4a+HIbWPy420+GYjcTl5FEjnn1ZmPtnjiuorl7/wAbeEfDHh7Sr661CCy0y8iT7AqQv80ewFdsaruChSvYAZAOMgUeHviL4R8VXhs9H1uCe6HSF1eJ34J+VXALYCknbnHfFAEXjv8A5lr/ALD1r/7NXW1yXjv/AJlr/sPWv/s1dbWUPjl8jur/AO60f+3vzCiiitThCis7XdZt/D+jXGqXaSvBBt3LEAWO5goxkgdSO9Yf/Cd/9Sp4o/8ABd/9lUSqRi7NnVRwVetDnpxutum/9M62ivMv+F6+D/8Ap7/8g/8AxyrFj8Z/Dep3kdnYWup3d1JnZDBHHI7YBJwofJwAT+FT7aHc0/s3Ffy/iv8AM9Forjbr4i29lbvcXfhzxHBAmN0ktiEVcnAyS2OpArsqqM4y2ZjXwtagk6kbX2+X/DoKKKKs5zkviZ/yT3VP+2X/AKNSt3XdT/sTw9qereT532G0lufK3bd+xC23ODjOMZwawviZ/wAk91T/ALZf+jUrM+Nv/JIdd/7d/wD0ojrJfxX6L9Tuqf7jT/xz/KmfLkEetePfGMUTzSXmrapcBWlkBPJ/iO0HaiqM8DCqvAwK+m7n4LeEYPCWpaZpmkQG/uLTy4ry7kd3Eyqdj7udnzYLbAAehBHFeEfBL/kr2hf9vH/pPJX1/WpwnwZpOmzazrNjpdu0az3txHbxtISFDOwUE4BOMn0NfedfBmk6bNrOs2Ol27RrPe3EdvG0hIUM7BQTgE4yfQ1950AFFFc//wAJ34P/AOhr0P8A8GMP/wAVQB0FFFFABRWH4l8YaB4Pt4J9e1KOzSdykQKM7OQMnCqCcDjJxgZHqKw7H4weAdQvI7WHxHAkj5wZ4pIUGATy7qFHTueenWgDuKKKKACisvUvEug6NcLb6prem2M7IHWO6ukiYrkjIDEHGQRn2NV/DXjDQPGFvPPoOpR3iQOElARkZCRkZVgDg84OMHB9DQBuUVXvr+z0yzkvL+7gtLWPG+aeQRouSAMseBkkD8ay5PGnhWFIXl8S6MiTJviZr+IB13Fcr83I3KwyO4I7UAblFFc/4k8ceGvCPljXNXgtJJMFYcNJIQc4bYgLbflI3YxkYzmgCl4g/wCSheDv+33/ANFCutrzKDxt4d8Y/ELwz/YGofbPsv2rzv3Mke3dF8v31Gc7W6elem1lT+KXr+iO7F/waH+B/wDpcwooorU4TkvEH/JQvB3/AG+/+ihXW1yXiD/koXg7/t9/9FCoNS+LXgPSrhYLjxLaO7IHBtVe4XGSOWjVgDx0znp6isqfxS9f0R3Yv+DQ/wAD/wDS5naUVy/h74i+EfFV4bPR9bgnuh0hdXid+CflVwC2ApJ25x3xXUVqcIUVycfxO8ES6pNpy+J9NE8Sb2dpgsRHH3ZT8jH5hwGJ6+hx1lABRRWfpmu6Prfm/wBk6rY3/k48z7JcJLsznGdpOM4PX0NAGhXnfxW+Ji+AdLgjsBaXGtXLgx285YhIud0jBcHGRtAyMkkjO0ivRK8z+KHw2h+JWl2Go6PeWkeoQoDBcNgxXML4OC6gnA+8pGRy3HzZAB558P8A486yNcttO8VywXVjdSiM3pRIXty2ApYjanlg5zkAjJOTjacP4wfDbUvDWqaj4nlvLSfT9S1NjEq7hKrS75MMuMYGGGQ3PBwM4HQeAPgd4m07xLo2uax/ZtvBa3AnktXlaSVShJX7mFzkKR85A4yDgqfQ/jrY2938KNSmnj3yWksE0B3EbHMixk8dfldhz6+uKAPkivuvw1ps2jeFdI0u4aNp7Kyht5GjJKlkQKSMgHGR6CvhSvv+gArkvEH/ACULwd/2+/8AooV1tcl4g/5KF4O/7ff/AEUKyrfD81+aO7Lv4z/wVP8A0iR1tFFFanCFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAHJf81e/7gP8A7Xrra5L/AJq9/wBwH/2vXW1lS+16ndjv+Xf+BHD/ABgmt4PhRr73Vr9pjMSIE8wph2kQI+R/dYq2O+3B4NfLHgT/AJKH4a/7Ctr/AOjVr6X+Os3lfCjUk+0wRebLAmyUZab94rbU5GG43dG+VW47j5o8Cf8AJQ/DX/YVtf8A0atanCfb9FFFAHlf7QUlinwyK3cMkk73sS2bKeI5cMSzcjjyxIO/LDjuOE/Zrs7d/EOuXrGf7VFaJFGFQmPY75bc2MBsomBkZG7g4OOr/aQnhXwLplu0sYnfU1dIyw3MqxSBiB1IBZQT23D1rD/Zl/5mn/t0/wDa1AH0BXxx8H/sf/C19A+3eR5Pmvt8/G3zPLfy8Z/i37dvfdjHOK+x6+IPAn/JQ/DX/YVtf/Rq0Afb9fGHxP8ADFx4V8falaywQQ29zK93ZrBjYIHdtgAGNuMFcYGNvGRgn7Pryf47eCrzxP4ctdTsDPNdaVv2WUFsZXuPNeJTjByNoUnofwoA6D4e+MLfVPhVY67f3OPsNoyX8jSmZ1MIIZ3wN25lUPjGfnHXqfDPhT4Kh+J3irWNU8QzySwQOtxcLE4jaeaVy2DhcBCFkztKkZGPbz+18R6vZ+H77QYL6RdLvnR7i2IDKzKQQRkZU5VclcZwM5wK+t/hX4Sm8GeA7PTryGOLUJXe4vAkhceYx4GemQgRTt4ypxnqQCf/AJq9/wBwH/2vXW1yX/NXv+4D/wC1662sqX2vU7sd/wAu/wDAgooorU4Tkv8Amr3/AHAf/a9eI/tB+EV0nxLa6/Y2UcNnqKFbholbBuQSSzcbQWUgjByxRyRnJPt3/NXv+4D/AO16yPjboDa98Mr5olkafTnW+RVZVBCAhy2eoEbOcDByB16HKl9r1O7Hf8u/8CMv9n7xDb6n8P8A+x1XZdaTKySDJO5JGaRX6YGSXXGT9zPGRXknxfmvvE/xB8RX9mJLrTNDSK1kmEexbcAhChJAyfOaTHUnkj5Rkcn4R8Y6p4J1G7v9I8gXVxaPa75k3+WGKncozjcCoxnI9Qa+n/ht4A/4Rz4avoGtR75tS82S/gEuVXzFCFAy4x8gUHBPzbsHGK1OE8//AGa9b/5DmgS3H9y9t4Nn/AJW3Y/64jBP0HWrHxad/H3xM0L4e2F59n8ndLdSuWKB2TfymAGZY1JBzz5uMrya8Qs77WfBPihprWT7Fq+nSyQk7Uk8twGjcc5U9WHf2r3v4B6BfTnV/HOrLG9xq7stvMGwzjzGMzFF+UBnC47/ACHAAIyAeoare2PgnwVcXUUUaWek2WIIXm2hgi4jj3tk5JCqCckkjqa+LLC6vJvENtefZ/7Uvnu0l8m4Qz/apC4O116vuPBHU596+z/Hf/JPPEv/AGCrr/0U1fHHhTWf+Ee8W6Tq5edI7S7jll8g4doww3qORnK7hgnBzg8UAfQ//C3/ABh/0SbXPzm/+MV5J41h+IPjzWYdU1TwdqUM8Vutuq2umXCqVDM2Tu3HOXPf0r67rH8VeIbfwp4X1HXLpd8dpEXCZI8xycImQDjcxUZxxnJ4oA5v4a6BfWfwf07RdSW7027kt51fy22TwCSR2VgedrhWB55B6jIxWf8ACn4Ur4BSfUNQuI7nWrhDCzQM3lRRbgdq5ALElVJJHYAAYJaRfi7Zw/C+PxpqGkz2v2iV7e1sxKJPPkBYDDgfKvyNksoI2tgN8u7yDwB4M1H4w65e6v4m12eS1tfklKzqZyzbmVUU5EcYJY/d29Qo6lQD1P4kfFXwha+FdU0m31SPUbzULKWCNbArMqeYjqGZwdoAPUZLcg4xXAfs2QQt4q1m4aK7M6WQRJFUeQqs4LBz1DkqpUdwr+ld/wCP/BXgjwv8K9Zki8N2iiC3KwTJGGnSV3AjbzWO8gOyk5Y/KCMEcVyf7Mv/ADNP/bp/7WoA+gK+ENC1P+xPEOmat5PnfYbuK58rdt37HDbc4OM4xnBr7vr4U8NabDrPirSNLuGkWC9vYbeRoyAwV3CkjIIzg+hoA7jQ7+7+MnxIsNP8XandrBIk32aKyRFWLCs+1c8KML94h2O1Qf7y/UejaHpfh7TksNIsILK1XB2QpjcQANzHqzYAyxyTjk18kfETwRefDXxbFHa3U7WsmLnT7wAo64b7pYADzEIGSvqp4zgfQfwp+Ji+PtLnjvxaW+tWzkyW8BYB4uNsihsnGTtIycEAnG4CgDX8P/8AJQvGP/bl/wCijXW1yXh//koXjH/ty/8ARRrrayo/D83+bO7Mf4y/wU//AEiIUUUVqcJ4P8d9bm0fUbiGDzA+o6THZmRJChRTOXbp1BWMoRxkMfofMPhPrHhrQfHMOoeKEzaxRMbeUxNIsE4IKuVHJwAwHBwxU4GMju/2jv8AkN6f/wBe0f8A6FLWJ8AdN0vWPGWo2Wq6TY38f9ntKhu4fN8srIg4U/Lzu6kE8DBAJzlS+16ndjv+Xf8AgR7P/wALt+Hn/Qw/+SVx/wDG6+fNL8T6RoXxpXX9JSODRU1N9u9CVS3clHZVVVIG1mZVxlflB3Y5+o/+EE8H/wDQqaH/AOC6H/4msfXk+G3gX7Hf6tpeh6bI0ubZ005DJvXncoRCwxx83Ykc5IrU4S58RvBi+OvB1xpCyRxXaus9pLJu2pKucZwehUsvQ43ZwSBXy58MfF3/AAhfjmy1KVttjL/o17xn9y5GW+6T8pCvgDJ24719n18yfHrw1D4Z8Y2HiPS7iS3n1R3mZIgI/Jmi2fvEZcHLFgx77gTk54APpO/vrfTNOub+8k8u1tYnmmfaTtRQSxwOTgA9K8L+E2lN8QPHmtfETWraN447jZZRPtYJKANvTGTHGEAJXksG+8tcp4z+LFx488DaN4at7O+OrySxi+dCNt06jaqqiD5t7EPtwNrKAA3BH0P4H8N/8Ij4L0vQzJ5klrF+9cNkGRiXfacD5dzNjIzjGeaAPO/2itfbT/B1josTSK+qXBaTCqVaKLDFSTyDvaIjH9089j5x8Ddf8KeH/Et3P4haOC8kRI7C8mUlISSVcE9EJDL8xGAA2WAOD0n7S9jbx6j4dv1jxdTRTwyPuPzIhQqMdODI/wCfsKn+BXgfw14h8G6lf6vpEF7dNdy2e+YsdsRjiPyjOFbJOHGGGeDQB3niP4l/DK8t7zQtZ1q0vLeVAk0ccMs0bAgEYeNSMjg5U5BHYivlnw3qU2la9b3ECxs7rJbkOCRtlRomPBHO1yR7469K+qv+FJfDz/oXv/J24/8AjlcT448LfCfwjdQWBs57TW5I2mtUilnkUkpIIy5YkbfMUdOdwXPy7qmfws3wv8eHqvzPYPC3/IoaL/14Qf8Aota1qyfC3/IoaL/14Qf+i1rWoh8KDFfx5+r/ADCiiiqMD5z+Kn/JPNW/7HOb/wBFPVT9nDTLO58W6nqM00BurO0C28DqC53thpFJORtC7SQP+WuMjOD6Mvgyx8deGvFWj30kkJHiC5mt505MMoAAbGcMMMQQeoJwQcEeK/Drxrd/C3xjfWmqQSfY2drfUraJEkkEke4KUO4DIYkHnBBPU4Iyo/AjuzL/AHqXy/JH0347/wCSeeJf+wVdf+imrwD9nH/koeof9gqT/wBGxV2/j34yeHdU8L3eh+GJJ9V1PVYms40jtJAq7yEIIbaxYqzbdob5gMj10Pgv8M7jwbZ3Ora3b+VrdzuhWMThxFBkHB2/LuZlyeW4C9DuFanCesV8kfGp3vPjHqVvcXnlwp9niR5izJAhiQngAkKCzMQoPU8Emvrevkj4631xd/FfUoZ5N8dpFBDANoGxDGshHHX5nY8+vpigD6n0K1s7Hw9plnp1x9osYLSKK3m3h/MjVAFbcODkAHI4NfMHx+/s7/haE/2L/j4+yQ/bvvf67Bx14/1flfd4/HNfUek6bDo2jWOl27SNBZW8dvG0hBYqihQTgAZwPQV8sfHlJl+Kl4Zb2O4R7eExRrKXNuuwDYw/gJYM+B2cH+KgD3/4ZWljN8JtDtV06SK0mststvdpnzC2fMYhico7FmHYqw4A4rxT48+C9C8K6jpF5otp9j/tH7QZ4UY+WGUoQVU/d/1hGB8oAGAOc+/+BP8Aknnhr/sFWv8A6KWvHP2mTN9o8NK0cYgCXJRw5LFsxbgVxgADbg5OcngY5AK/wF+H2i67ZXfiTWII74wXDWkNnPGGiU7FYuwPDHD4AIwME8nBXb+O3gjQrHwNDquk6NY2Fxa3aB3tLUx7o3BUg+Wu372w5fAGCActhuo+BX2P/hVGm/ZvI87zZ/tXlY3eZ5jY34/i2bOvO3b2xWP+0d/yTzT/APsKx/8AoqWgDzT4J/D/AEjxtqmp3GteZNaackY+yqxQStJvwWZSCAuwnAxkkc4BB9v8d+DfDI+HOupHoGmwi1sri6g8i2WMxSiPdvUqAQSY0zjqFAORxXnf7Mv/ADNP/bp/7Wr2TxpIsPgXxDK8Mc6JplyzRSFgrgRN8p2kHB6cEH0IoA+aPgNqU1j8VLO3iWMpf281vKWByFCGXK89d0ajnPBP1H0v4l8H6B4wt4INe02O8SBy8RLsjISMHDKQcHjIzg4HoK+XPgl/yV7Qv+3j/wBJ5K+v6APhTVtEm07xVfaDb+ZeT297JZx+XGd0zK5QYUZOSR05696+s/D3wj8FeHrMwJo0GoSP9+fUkW4dsEkcEbV64+UDOBnJGa+WPAn/ACUPw1/2FbX/ANGrX2/QB8cfF+ws9M+KWs2dhaQWlrH5GyGCMRouYIycKOBkkn8a9n+Enwu8PxeC9P1jWdFgvNTv4jK32wLMixMcx7U5UZQK2SNw3MMjoPIPjb/yV7Xf+3f/ANJ46+n/AAJ/yTzw1/2CrX/0UtAHz58cvAmgeDLjRJdBtZLVLxJlliMzSLlCmGG4kgneQeccDgc59f8Aghqq6n8K9MT7TJPPZPLazb9xKEOWRMnqBG0eMcAYHbA4P9pr/mVv+3v/ANo10H7OP/JPNQ/7Csn/AKKioA80+N3j6bxP4lk0KCKSDT9HuJIiGc5nmB2s5AOMDBC98Fjn5sD1/wCHXw48EQ+DtPvItP03WZbu3R5r2ZBcK787tgcfIAxZcbVPygNyK+UJ4JrW4lt7iKSGeJykkcilWRgcEEHkEHjFfTf/AAzj4P8A+glrn/f+H/41QBzHxx8AeEfDmh2+q6THBpupTXZ/0VZXxcqeW2R8hdp2njaoBI6lRXX/AAa8Q2/jb4az+HtTXzpLCL7BcJkr5ls6kR8qBj5QycEn5Mk5NV/+GcfB/wD0Etc/7/w//Gq7zwV4K03wHo02l6XPdzQS3DXDNdOrMGKquBtVRjCDt60AfKnxQ8MW/hH4gajpdjBPDYDZLaibJyjKCdrH7yhtyg8/dwSSDXafD/4IQ+MfBH9uXesSWk925FmIow6xqjlWMgOCxJUgAEY4OTnAp/tDWNvafEqOaCPZJd6fFNOdxO9wzxg89PlRRx6eua9v+D9jcaf8KNAhuo/LkaJ5gNwOUkkeRDx6qyn2zzzQByfgb4R6v4C0vWtWh1KO58ST6ZNb2kNuo8qOQ/Mp3SY3EssfUKB8wORzXhnw7vdC0/x3pt14lSB9ITzftAngMyHMThcoAc/MV7cda+16+ZPif8EpvDdvda94daS50tHLzWZUmS0jwDkNkl0Bzk9VGM7sMwAO/wDif8LtN8WeGo9V8KafaHVkSD7O1qyolzbgbQo+YRgBWBDcnCBRxjEXwG0q+0Wy1aw1K2ktbtEt3eGThlD+Y65HY7WHB5HQ4PFfOugeI9X8LaoupaLfSWl2EKb1AYMp6hlYEMOhwQeQD1Ar6a+EPiWbxfd69rtxbx289ylqJI4ySu5FdCRnkAlc45xnGTjNZVPij6/ozuwn8Gv/AIF/6XA9RooorU4QrhvijpsOs6NpGl3DSLBe6tDbyNGQGCurqSMgjOD6Gu5rkvHf/Mtf9h61/wDZqyrfAzuy3/eo/P8AJnyz8Q/A83gDxKuky30d6klulxFMsZQlSWXDLk4O5W6E8YPfA7z4W/BrR/Gfhdta1bVp/wB5K0ccFhIgaHacHzSyt8x4IUYwpByd2Bl/tBMx+JpDX8dyBZRBYlVQbUZb922OSScvk84kHYCva/gl/wAkh0L/ALeP/SiStThOk8KeEtI8GaMNL0aGSOAv5kjSSF2kk2qpc54BIUcAAegFfHnjv/kofiX/ALCt1/6Navt+viDx3/yUPxL/ANhW6/8ARrUAe7+BfgJo1poYl8YWv23U58OYUndEtR/cBRhubnk5I7DpubzD4x+ALPwL4htP7KjnXTL+JpI/OlD7ZA53Iv8AFtVWjxuyTnqecfW9fOH7Smp+b4h0PSfJx9mtHufN3fe819u3GOMeTnOed3bHIB6n8GtVbVvhXozy3Mc89uj2r7duYwjlURgOhEezryQQec5rwD4qfDOH4d3Gntb6pJeQag85jSSEI0SoUwCwOGOH64Xp05497+CX/JIdC/7eP/SiSvP/ANpr/mVv+3v/ANo0AcB4F8E+LPiLZjTbfUZ4PD1lKCzXMrmCNyckRx9Gkw7NgY68kbhnvPhZ8LPGnhT4kQ39/BHb6bbpMk00d0pW5UqVUBVO4gttfDAfdycEAV1/7PumzWPwyFxK0ZS/vZbiIKTkKAsWG467o2PGeCPoPVKAPkD4ueOrjxn4tmhBg/szTJZYLLyWDiQbsNLvH3t+1SMcAAYzyT6/4Y/Z88NafZwyeIGn1S+MWJ41maOBXJzlNu1+BxktzycDIA8A8Cf8lD8Nf9hW1/8ARq19v0AfNnxX+DFj4X0SXxDoE8i2cDlrq3uptxQO6LGsWFyQCxzuYnAHJOaufAjWf7e0PXvAGoPObWa0llhkQ8xRviOVQSSBy6soC4yXJ616P8bf+SQ67/27/wDpRHXin7Pumw33xNFxK0gewspbiIKRgsSsWG46bZGPGOQPoQDl/iH4Hm8AeJV0mW+jvUkt0uIpljKEqSy4ZcnB3K3QnjB74Huf7OP/ACTzUP8AsKyf+ioq4D9o7/koen/9gqP/ANGy13/7OP8AyTzUP+wrJ/6KioAw/jj8M4Smr+PItUkVwkAls2hDBm3JDlXyNo27Tgg8g888cp+zza/aPiVJL9oni+zafLLsifasuWRNrj+Jfn3Y/vKp7V7P8bf+SQ67/wBu/wD6UR14J8Dp5ofi3pCRSyIkyTpKqsQHXyXbDeo3Kpwe4B7UAfS/j7xLN4Q8Eanrtvbx3E9siCOOQkLud1QE45IBbOOM4xkZzXyx8PPCE3xI8atZ3moSImx7y9uGYtLIu5Q20nOXZnHJ6ZJ5xg/Rfxp0T+2/hfqmy3864sdt7F8+3ZsPzt1AOIzJwc+wzivGP2eb63tPiVJDPJsku9PlhgG0ne4ZJCOOnyox59PXFAHb65+znokeh3smiX+qyamkTPbRzyxMkjjkIflXG7pncMZzzjFeMfDvw3Z+LvHem6HfyTx2t15u94GAcbYncYJBHVR2r6z8e+I4fC3grVNSe+jtLgW8iWbsAxa4KnywqkHcd3OMEYBJ4Br50+AmjXmofEy21GFP9F02KSW4kIOBvRo1UEDG4lsgHGQrelAH0H4H+Hmi+ALe8i0l7uV7x1aaW6kDMQoO1RtAAA3MemfmOSeMfMnxgsbfT/ivr8NrH5cbSpMRuJy8kaSOefVmY+2eOK+x6+MPipqf9r/FDxDc+T5Wy7Ntt3bs+SBFuzgddmcds4560AemeHPhXq/xPt7Pxf418QXebtCEto4AknkgEIQSAqAnLYCEEHOcsSPO/ih4AT4f+IYbK3v/ALZa3URniL7RJGN7Dayg5OAB8+FDHOB8px9j18wftHf8lD0//sFR/wDo2WgDq/A2s/2v8L/BsTPPJNp/iOGzkeY5zgs6hTk/KEdFHTG3GMAV7pXzX8KLG3j8G2F+seLqbxdawyPuPzIkeVGOnBkf8/YV9KVlD45fI7q/+60f+3vzCiiitThOS+Jn/JPdU/7Zf+jUryX9oHx5cNqP/CHafPttUiV9RGwHzHJWSNOVyNoVWyp534PSvWviZ/yT3VP+2X/o1K5z4peBfAV9t8S+Kb2fSWXbDJcWjANcE8KCuxi7AA9BnaDnhRjJfxX6L9Tuqf7jT/xz/KmeYfDD4OWfjfwleavqV7fWcjytDZeXGAh2qP3h3D94u44wpX7jDOTx534w8NTeD/Fl/oM9xHcPaOoEyAgOrKHU4PQ7WGRzg55PWvY9I+Nvh/w3o+k+HPB/hjVb+OPMSx3cyxyO7NkY2CTezMzHAC4JAAxwPHPGGual4j8WX+qaxax2uoSOqTwJGyCNkUR42sSQfl5BPXNanCfRR1++8Ufs9LrGptG15OirK6LtDlLkJux0BIUE4wMk4AHFes14X4W+2f8ADMr/AGnyPJ80/ZfKzu8v7UM78/xb9/Tjbt75r3Ssl/Ffov1O6p/uNP8Axz/KmFFFFanCcl8TP+Se6p/2y/8ARqV4L8UPDXxOP9o6r4mm+06RFKkpNteA2kZbCL5cLEMMb9uduepJOST718TP+Se6p/2y/wDRqVmfG3/kkOu/9u//AKUR1kv4r9F+p3VP9xp/45/lTPljwtZa7qHiO0tfDTzpq77/ALOYJxC4wjFsOSMfKG789K9Un+GvxjuvD8stxr13MZUKSaZJrEjSOpO0g5PlEEc439PfiuT+CX/JXtC/7eP/AEnkr6z1bUodG0a+1S4WRoLK3kuJFjALFUUsQMkDOB6itThPijwXG03jrw9Ek0kDvqdsqyxhSyEyr8w3AjI68gj1Br7P8R6/Y+FvD95rWpNILS1QM/lruZiSFVQPUsQOcDnkgc18aeBP+Sh+Gv8AsK2v/o1a9j/aZnmW38NW6yyCB3uXeMMdrMoiCkjoSAzAHtuPrQBxj6x41+Nvi2602yvPslnJEZDYtcsltDCjDbvA/wBY24r820nJHAUfL0Gs/s3apaac82ka7BqF0mT9nmt/s+8AE4VtzDcTgAHA55IxXn/w/wBX8a6JqM954Ptb66xsF3DBatcRuuSVEigHGcMARhsFsEZNemQeOvjZqtvFpVt4Ykt7t0Ci+k014WyoyWLSnygWweoA5wADigDL+A3xAuNN1yLwpqN1nTLzcLMSEYgnPIUMSMK/I285crgDc2fR/i98Uv8AhB7NNK0xd+u3cXmRu6ZS2jJK+Yc8M2QQF6cZbjAbyDwd8JPGw8W6Dc3GmT6bb+al4bt2jzCiMrHIO7bJyMI68nqMK2I/jzqU198VLy3lWMJYW8NvEVByVKCXLc9d0jDjHAH1IBX8D/DzXvirf3mpXeqyRwROqXF/dh5pJG2HAXPDkBUBBYYDL14FaHxA+Cd94J8P/wBtW+qx6laROFusweS0QYhVYDc24Fjg9xkcEZIp/D7/AIWfa+HtQvvBPntpglP2hU8mT94iAnbHJli20r90Zbgc4AHQX1h8cfGfhySzv7SebTbvG+GeO1tnO1wRlTtdfmUHtn6GgDp/2fvHVxqlnP4TvzPPNZRGe1ndgQsAKr5R7/KWGOvBxwFAOh8ZfirceEvI0Tw9dQDV5P3lzLgSNapwVG0grufPfOFGcfMpHOfBj4V6/pPiyHxHr1lJYQW9uXtY3kXfI8ilfmQZKgKzZDbSCV64IryPx3/yUPxL/wBhW6/9GtQBueCvhp4j+JFxNqKTxx2ZuGW61C6l3sZMqz4XO5nw+7nAPOWBq543+G3iD4WT22s2mr+ZatL5MF9aO0E0blCSCoOVyN4+VjwDnGQK+l/An/JPPDX/AGCrX/0UtY/xY8HXHjbwNNYWPN/byrdWqFwiyOoIKkkd1ZsdPm25IGaAPGJv+Ei+Nvgu2MP+la/4dlYXCv5cYvI7g5VlI2qrJ5WCpABAzuz8teP19T/Bn4d+IPAv9pTavNYiPUIoT9nhLPIjruOGbhRjeQQN2T0IA+byj46eEf8AhHPHL6lbrix1ndcpz92YEeavLEnkh84A/eYH3aAPZ/gd4euNA+Gts9037zUpTfhMD5EdVCcgnOVRW7Y3YIyK8Q+Nnhq40Dx9PdXGpfbv7V33alyA8ILsBGV3FtqqFAYgA4wB8px6H+zl4oWfS9Q8LzySGe2c3ltudmHlNhXVRjChWwevJlPHBNeWeP8AxDcfEf4lO+nL50csqWGmpkLvTdhOWC43Mxb5um/BOBQBp/BnTLy88X2LxTT2kbXaMtwinD+UDJJGDkZyuFIzwJOQQcH60ry7QNAbwvN8NtHlWRZ4Le8adHZWKSvHvkXK8EBmYDGeAOT1r1GsqfxS9f0R3Yv+DQ/wP/0uYUUUVqcJ5d8XvDU3i+70HQre4jt57lLoxySAldyKjgHHIBK4zzjOcHGK+Zdf8Oav4W1RtN1qxktLsIH2MQwZT0KspIYdRkE8gjqDX194g/5KF4O/7ff/AEUK4D9pDRJrvw1pGsxeYyWFw8UqLGWAWUD52b+EBo1Xkclxz65U/il6/ojuxf8ABof4H/6XM8c+Femf2v8AFDw9bed5Wy7Fzu27s+SDLtxkddmM9s556V9B/GrwZr/jLw1ZwaFJHKbW486SxbapnJG1WV2IAKgtwSAQx5yAD8saTqU2jazY6pbrG09lcR3EayAlSyMGAOCDjI9RX0n8cPGkVv8ADWyi0q7/AORg2mNwrqz2u0OxB4xndGpDdVdhjrjU4T5s0rSr7XNUt9M0y2kuby4fZFEnVj/IADJJPAAJOAK+n5NdtPgh8MNJ0vUZY9Q1RUcwW8e+NZ2MoaQB9rABBL1YDdjoM4HL/s5eFJoU1DxXcCREmQ2VqpyA67g0j8jkblVQQeocEcCuI+POpTX3xUvLeVYwlhbw28RUHJUoJctz13SMOMcAfUgHP63r3i/4nazvmiu9TngR2itLK3Zlt4y3O1FBOMlRuOScKCTgVHr3gnxd4F+x6jqunz6ful/0e5imRtsi/MPmjY7W4yM4JwcdDiPwh40vvBlxfS2dlpt4l7bm2niv7fzVZCeRwQcHoRnB7g4GOs13416p4n+H954a1fTYJbq48ofb4X8vhGRstHggsShyQVHzcAY5APQ/g38XLvxHfyeHvElxJPqk7tLZ3IiRVdQmWjIRQAQFZgT1yRkYAPkHjO5+Iel6oieK7zWYZxcfaoPNuGMQlGG3wlTsBXePufdzjjpUnwhTVD8UNFl0mz+0yRSkz5GVjgYFJHJyMYVzjJ+9tGCSAfR/2mv+ZW/7e/8A2jQBkfs7395b+IdX8+7ni0S20+SeffIVtopC8eHbPyq2xG5POFPYGvV/jb/ySHXf+3f/ANKI68g/Zx/5KHqH/YKk/wDRsVev/G3/AJJDrv8A27/+lEdAHyZpOmzazrNjpdu0az3txHbxtISFDOwUE4BOMn0NfedfAFff9AHzJ4m+PmtTeLBNo1vpraXYXEhtFnty5nBUoJGJwynBYgLsIDlW3c57rwd41m8eal4N1S7gjhvIrjULe4WJCsZYRKwKZZjja6Zyeue2K8d8XfCnxN4f8S3djYaLqWoWG8va3FtA04aIk7dxRcBwOCMDkZHBBPqfw18L33hY+CodSjkhu7241C8e3kTa0IaBFVTyedqBucEbsEZFZVvh+a/NHdl38Z/4Kn/pEj3CiiitThCiiigAooooAKKKKACiiigAooooAKKKKACiiigDkv8Amr3/AHAf/a9buu/2j/wj2p/2R/yE/skv2P7v+u2HZ975fvY68etYX/NXv+4D/wC1662sqX2vU7sd/wAu/wDAj5MvPg78TdUefUNQ0+S5vGdFY3GoRSSyjaRu3FyMKFUcnPK4BAONCD9nXxlNbxSvd6NA7oGaKS4kLISPunbGRkdOCR6E19R0VqcJ438HfCnj/QdZuH8Uz3celxWQt7a0m1DzlDbl2lEVmVQqoR2+8AM849koooA+ePEXwT+IniHUZJdR8U2OpRrLI0DXdxMNoY5JEYQrHnA+VTgYAHAFcnpvwo+KujXDXGl6dd2M7IUaS11OGJiuQcErIDjIBx7CvrOigDyfx54W+J2u6HBZadrtj9nOlRxalbqwje7uRkybG8sYV+BjcoPIIAJrxj/hSXxD/wChe/8AJ23/APjlfX9FAHP+Df8AhKf+Eei/4S/7D/afH/Hpn7uxf9Z/D5m7dnZ8vTFdBRXjfx1+INjpnh+68J2c8j6teoon8mTb9miyGIcjqXUbdn91iTwQGAPMPgZ4XXxD8QYbu4jkNppKfbCQjbTKCBEpYEbTu+cdc+WRgjOPrOuL+GHgmHwT4OtbV7eNNUuEEt/IANzSHJCEgkEIDtGDg4JAG412lAHJf81e/wC4D/7Xrra5L/mr3/cB/wDa9dbWVL7Xqd2O/wCXf+BBRRRWpwnFXuoWWnfFgTX13BaxNoe0PPIEUnz84yT14P5Vv/8ACU+Hv+g9pf8A4GR/41YvdF0rUZhNfaZZ3UqrtDzwK7AdcZI6cn86r/8ACLeHv+gDpf8A4Bx/4Vio1It2selOthasYe0UrpJaWtofM/8Awr3Sf+Fw/wBmfb7L/hFvN+1/aftK+X9n+95O7zM7s/u/vb8fPjFfTH/CU+Hv+g9pf/gZH/jR/wAIt4e/6AOl/wDgHH/hR/wi3h7/AKAOl/8AgHH/AIU/3vkZ/wCw/wB/8D53+JPgq31n4oJeaTqeljStWliM9xDcQBbVyCJGZfMBPCGQscbmfbksRn6AsNc8K6Zp1tYWetaXHa2sSQwp9tQ7UUAKMlsnAA61Y/4Rbw9/0AdL/wDAOP8Awo/4Rbw9/wBAHS//AADj/wAKP3vkH+w/3/wD/hKfD3/Qe0v/AMDI/wDGvm/4peANOTxCuo+Dp9Ll0+72+bbQ6hCPImZ9vyqWGIzuU8ZC4fO1QK+kP+EW8Pf9AHS//AOP/Cj/AIRbw9/0AdL/APAOP/Cj975B/sP9/wDA+W/DnxQ+IXhq3s7O3llubC0QpHa3dpvXbg4BYAPgZ4G7jAHQYrY0uy1b4s+IF1Px1r1tp+k2rvGsDXKW0ihgWxDGwPAbYCz8kADcxXj6M/4Rbw9/0AdL/wDAOP8Awo/4Rbw9/wBAHS//AADj/wAKP3vkH+w/3/wPP/iTpXh7xH8NU0DRda0tJtN8qSwgOoxhW8tSgQsxOfkLAZI+bbk4zXg/gjxX4m8A6jc3mk2Jl+0xeVJDcxSNGcEENtUr8w5AJ6Bm9a+uP+EW8Pf9AHS//AOP/Cj/AIRbw9/0AdL/APAOP/Cj975B/sP9/wDA+X/GXijxh8TfNlntYbPTNPzKlj56xdd2GxIwaaQKNvyj6KN3PF6bL4g0a4a40t9TsZ2Qo0lqZImK5BwSuDjIBx7CvtT/AIRbw9/0AdL/APAOP/Cj/hFvD3/QB0v/AMA4/wDCj975B/sP9/8AA8n8dfEbxZo3hLQho11pdxfT2ka6jcwSxXE8VztUsBGuUCkhhuAYc4+X5S3z/YR6vpmo21/Z21xHdWsqTQv5BO11IKnBGDggda+2P+EW8Pf9AHS//AOP/Cj/AIRbw9/0AdL/APAOP/Cj975B/sP9/wDA4/xNdeHviJ8NTaXmpaXZX15aR3MMU95HutLjaGVSSMjBJRiFB2lhxmvl+xTX/DeuR3lgtxb39lKdk0A3gMMg4YZV1PI7qwPcGvtD/hFvD3/QB0v/AMA4/wDCj/hFvD3/AEAdL/8AAOP/AAo/e+Qf7D/f/A8/+DGt6l4ibW9U1cAX8qWqzER7NxRXQMV7EhQTjAyTgAcV6tVSx0rTtM8z7BYWtp5mN/kQrHuxnGcDnGT+dW6qnFxjZ+f4szxlaFarzQWlorX+7FL9AoooqzlPLvH3hqbxf4m1PQre4jt57nw+hjkkBK7ku1cA45AJXGecZzg4xXzzp8/iD4YeNLe9uNM+z6nZ+YUhvY22OGDxlhgjcvLYZTg46mvqr/mr3/cB/wDa9bus6HpfiHTnsNXsIL21bJ2TJnaSCNynqrYJwwwRng1lS+16ndjv+Xf+BHj9r+0po76dO954fvor4bvJhhmSSN+Pl3OdpXJyDhWwOeelcxoek+LPjd4tstZ8Rw+R4etdrYEbxwOgbDRwjOWZypDMG+XHJ4Va9j034S+A9KuGnt/DVo7shQi6Z7hcZB4WRmAPHXGevqa7StThCvL/AI7+GLjxB4B+1WUEElxpUpu3Z8BxAEbzApP/AAFiMjOzuQAfUK8b/aK19tP8HWOixNIr6pcFpMKpVoosMVJPIO9oiMf3Tz2IBwH7P/hSHXPGM+s3QjeDR0V0jbB3TPuCHBBGFCu2cghghHevqOuH+E/g648E+BobC+4v7iVrq6QOHWN2AAUEDsqrnr827BIxXcUAed/F/wCH83jvw1CdP8v+1tPdpbYOxAlVh88ec4BbCkE91AyASR4J8O/iJqnw21yW0u4Z5NMeUpfWDja8bj5S6g42yDGCDjdjBxgFfr+sPX/B3hzxSjLrWj2l25QJ5zJtlVQ24BZFw6jOeAR1PqaAPN5/2kPCq28rW+lazJOEJjSSOJFZscAsHJAz3wcehrz3Q/Ceu/Ei61r4ha+/2a0ijkuI2jQJ9okjTCIgP/LNdqgsck4xktuZfXtM+BngbTdRlvGsJ7zdKJYobuctHBgk7QoxuXkDD7sgD3z2Xin/AJFDWv8Arwn/APRbVM/hZvhf48PVfmHhb/kUNF/68IP/AEWta1ZPhb/kUNF/68IP/Ra1rUQ+FBiv48/V/mFFFFUYHnVjfXGmeDfiDf2cnl3VrfajNC+0Ha6xgqcHg4IHWvDfhH4t8J+FdRu5PE2mfaZJpbc2l19lSb7IVLbnyTuXqp+QE/L0yBXX+J/itfeCbjXdF0a3j/tC51a6ne6mXcsSMdi7Fzy+5Sfm4GBw2eFsP2a7i4062mvPEn2S6kiR5rf7CJPKcgFk3CXDYORkdcVlR+BHdmX+9S+X5I9D8NePfhXDcTwaDf6NpryIHlItfsSuFOBlmRAxG44Gc8n3rc8PfEXwj4qvDZ6PrcE90OkLq8TvwT8quAWwFJO3OO+K8c1L9mnUordW0vxJaXM+8BkurZoFC4PIZS5JzjjHc88c8X4u+Dvinwbo7atemxurGPHnS2k5PlZZVXIcKTksB8oPQ5xWpwn1fret6d4c0efVtWuPs9jBt8yXYz7dzBRwoJPJA4FfHHxE8SWfi7x3qWuWEc8drdeVsSdQHG2JEOQCR1U96+m/hp41h+JHg6Z9RgtDeI729/aKg8sq2dp2MzHYynHzdSrjoK3P+EE8H/8AQqaH/wCC6H/4mgDk5vjz4Di0sXaX13NOUVvsSWjiUE4yuWwmRnn5scHBPGfBPir43s/Hvi2PU7C1nt7WG0S2QTkb3wzMWIBIHLkYyeme+B9D/wDCkvh5/wBC9/5O3H/xyj/hSXw8/wChe/8AJ24/+OUAYfwi+KOm65o0Ph+4tJLCfR9MVpLiSVTA0MKxoXLHBU5OcYIAH3q8w+NnxA0jxtqmmW+i+ZNaackh+1MpQStJsyFVgCAuwDJxkk8YAJ+j9E8HeHPDlx9o0fR7SznNulsZIkwxjU5AJ6kk8ljy2BknAxzf/Ckvh5/0L3/k7cf/ABygDk/gB4u0U+H08Km9uxqyvJcCK6YeWwJOVg56BQGKnByzkZGcZ/7Q/i3SLrTbfwrBNJJqltepcXCCMhYl8psAscAkiVSNuehzjjPceEvgx4Z8HeIItasp9SuLuFGWL7VMpVCw2lgFVcnaSOcjk8ZwRoar8JvA+tapcalf6DG93cvvldJ5YwzdztRgMnqTjk5J5JoA8Y+AnjDQPC1xrUGt6lHZPfvbJbmRG2sQZAcsAQoG5eWIHPsa9b+LHjTQvD3hLVNJvrv/AImWo6fNFbWsalnbepQMeyrk5ySM7WxkjFH/AApL4ef9C9/5O3H/AMcq544+GWi+P7izn1W61KF7RGRBazgKQxBOVZWAPHUAE8ZzgYAPljwD4lh8IeN9M124t5LiC2dxJHGQG2ujISM8EgNnHGcYyM5r631bx94X0LRtP1bUtWjgs9RRXtGMbs0qlQwIQKWxgjJI4yAcEisP/hSXw8/6F7/yduP/AI5VzxV8LvDnizRtK0u7W7toNKTy7RrWbDJHtC7DvDAjCpyRn5RzycgHy5/bGl6f8U/7bs0zpFvrf2uFLeLZmBZ96hEOMfKBgHGOnFfYc/iTRbbw/Lrz6paHSY0LteRyh4yAdvBXO47vlwMkngc8V5Xffs3eG5LORbDWdVgujjZJOY5UXkZyoVSeM/xD156V2mgfDmx034eL4N1i6k1iw3lm3L5Ax5nmBV2HcAG55YnJPOMKAD5g+Jev2Pij4h6vrGmNI1nO8axO67S4SNU3Y6gEqSM4OCMgHivp/wCF3iPSNd8C6Vb6bfRzz6dZW9tdxgFWikESggggHGQQGHBwcE4NY+m/AbwHY27RXFjd6g5csJbq7dWAwPlHl7BjjPTPJ56Y6jwl4E0DwSl2uh2skJu3DTM8zOWAZii8nAChyB3Ixkk80AfOnx61vTtb+Ia/2dced9htBZXHyMuyZJZdy8gZxkcjI969H/ZuvreTwbq1gsmbqHUPOkTaflR40CnPTkxv+XuKuT/s6+DZriWVLvWYEdyyxR3EZVAT90boycDpySfUmug8GfCPwz4I1R9TsBd3V4U2Ry3rq5hBzu2bVUAkHBPJxwMAnIB8+fF/wjqPhrxzqF7crusdVu5rm0nyo35Id12hiRtMm3JxnGRXs/gL43eH9b060sdevf7O1dIlSaa72pDO4B3OHGFXO3OGC8sFG6vTNV0qx1zS7jTNTto7mzuE2SxP0YfzBBwQRyCARgivG9T/AGa9Hl8r+yfEF9a4z5n2uFLjd0xjbsx365zkdMcgHQeNPjd4a8P6ddRaNewatq64SKKHc0IJGd7SD5WUdwpJJ445Iz/gz4WfwJ4N1LxLq99B9l1G0hvsQqzeTAkbSZbjJbDnKgHG3gnPEmlfs8eELJ7eW/udS1B0TEsbyrHFK23BOEAdRnkDf2GSec+ia34Y0vXvC8/hy6g8vTZYliEdv+78oKQU2Y4G0qpAxjjBBHFAHyh8WPGNv428czX9jzYW8S2tq5Qo0iKSSxBPdmbHT5duQDmvc/g58QPDmreGtK8NQeXYataW/lGzK4E2wAtIjAAMWyWI+9necEDcaf8Awzj4P/6CWuf9/wCH/wCNVueEvgx4Z8HeIItasp9SuLuFGWL7VMpVCw2lgFVcnaSOcjk8ZwQAaHxZ1W+0X4Ya1f6bcyWt2iRok0fDKHlRGwex2seRyOowea8z+B3xPeW8uPDniXVZ57i6lD2FzeStIWcjBiLsxxnClRjklhnJUH2jxT4bs/F3hy70O/knjtbrZveBgHG11cYJBHVR2rxC+/ZovI7ORrDxPBPdDGyOezMSNyM5YOxHGf4T6cdaAKf7Qc/hC51S1fTZY5fEgcpetasrII1yuJcf8tQwAA6gAhuNldP+zl/yA9U/7Zf+hS1kaB+zZMXWTxHrkaoHIMGnKWLLt4PmOBtO7qNh4HXnj0L4e6BY+FvEXiXRdNWQWlqlmqeY25mJjZmYn1LEnjA54AHFZVPij6/ozuwn8Gv/AIF/6XA9BooorU4QrkvHf/Mtf9h61/8AZq62uS8d/wDMtf8AYetf/ZqyrfAzuy3/AHqPz/Jnz78ftM+wfFCe587zP7QtIbnbtx5eAYtuc8/6rOePvY7ZPofwT+Ivhy18EW2gapqVpp15YvLt+1S7FljZ9+4MwCg5kK7ck/KT06dp8R/hrZ/ESzs1m1CeyurLf9nkRA6fOU3blOCeE4wwxnPPSvMIf2aLxry5WbxPAlqu37PIlmWeTj5tylwEwemGbPXjpWpwnuem+JNF1fRm1iw1S0n09ELyXCygLEAoY78/cIUgkNgjvivizxZfW+p+Mtcv7OTzLW61C4mhfaRuRpGKnB5GQR1r6L1/4IQ3Hg6w8N6BrElnZw3pvLkXcYma4kbam8sMbSke4BQAG4zg/NXN/wDDMv8A1N3/AJTf/ttAHumlarY65pdvqemXMdzZ3Cb4pU6MP5gg5BB5BBBwRXzB+0BqtjqfxGRLG5jnNlZJa3GzkJKJJGKZ6EgMM46HIPIIHqfhH4IW/hyz1u0u/EV9d2+q2n2WSO2U2oUZzuOGbcw6DPGGcEMGIrl/+GZf+pu/8pv/ANtoA7D4CazZ6h8M7bToX/0rTZZIriMkZG92kVgAc7SGwCcZKt6VwH7SOs2d3rmi6RC++6sIpZbjBBCebs2qcHIbCZIIHDKec12fgT4Gw+DPFltr0uvyXz2yOIoVtRCNzKUyx3tkbWbgY5xzxg8g37NOpb70L4ktCiIDZk2zAzNtORIM/uxuwMjfxzgdKAOr/Z51xLvwNJpE9/A91aXcpgtd6iRICEYttHJXe7fMe5xnoK9gr580r9nvXtK8WW9zB4njg0+J8m8tGeG72lcMFUAhSclc7zwc4P3a+g6APijxp4bvPAPjm606KSeL7NKJ7G5DFXMZO6Nw2F+YdCV4DK2OlfYfhzX7HxT4fs9a01pDaXSFk8xdrKQSrKR6hgRxkccEjmsfx58P9I8faWtvqHmRXcCOLO6RjmBm25O3IDA7VBB7ZwQea8Em+DnxO0H7RZaS3n2t5EBcnT9REUco+YbHVyhbgnsRhuvWgDt/2g/GlnDoa+ErO73X88sct7EihgkI+ZVYn7rFtjADnC84BG7P/Zu8N/8AIW8UPJ/1D4o1b/ckcsMf9c8EH+9kdKZ4R/Z1mL2l74q1CNUDh5dNtgWLLtB2tLkbTuyDtB4HDc5H0HQB80ftI2NxH4y0m/aPFrNp/kxvuHzOkjlhjrwJE/P2Ndv+zj/yTzUP+wrJ/wCioq7D4ifD+z+IOhxWM1x9jureUS292IRIU7MpBwSpHUAjlVPOMV4BdfA34haPqMEmnW8F3JHtlS6sb1Y/LcHjBkKMGGAcgdxznoAex/HfVbGy+GF/YXFzHHd37xJawn70pSVHbA9Ao5PTkDqQD4x8CNVttM+Jtul1cyQC9t5LWPGwI7khlRy3IBK8beS2wdCQeg0X9nvxHrSTX/iTWY9Ou5nLlCv2uVmLNuMjBwMng5DNndzgirlj+zReSWcbX/ieCC6Od8cFmZUXk4wxdSeMfwj0560AfQ9fIml+Grn4kfEHVbvwPaR6HaQOLqEyyOiwEEBcMitsdmy4UcLggHC19L6rYaLofw5uNK1O+kttFt9M+wy3LsN6xeX5ec4wXIxgBeSQADnFfFEE81rcRXFvLJDPE4eOSNirIwOQQRyCDzmgD6Dtvgj4w1zbB4y8bz3FjHKkiwRXE1zv6huZdoRsHAbDfePHY+ueGvB+geD7eeDQdNjs0ncPKQ7OzkDAyzEnA5wM4GT6mvliDx98S/DFvFcSatrMMGoIHgk1GMzLKoGcxmZWGMOCSvXIz2rU0b49+NtP1FJtRuoNUteA9vNBHFkZBJVo1BDYBAJyBnoaAPq+viDx3/yUPxL/ANhW6/8ARrV9F6+y/G/4ZMPCt/JZFb0CWK9Vow5QcxybNwxh0cEbhkL0OdvmE/7OvjKG3llS70ad0QssUdxIGcgfdG6MDJ6ckD1IoA+l9JjvodGsYtUmjn1BLeNbqWMYV5Qo3sOBwWyeg+gr5s/aO/5KHp//AGCo/wD0bLXpfws8C+L/AAp4a1Wy1TV44Hu7dBp6JK1wNPkIkLHYwCZ3OpIUkMVPPevLL74IfEXU/Eci37wXbSY36rPfeYjYQYzn96cYC/c7enNAF/4T6lDL4Vs9LVZPPt/FdpcOxA2lZEKqBznOYmzx3HXt9MV8t+D/AId+KvCPjLRb/XNK+yWsl9BCr/aIpMuZFIGEYnop/KvqSsofHL5HdX/3Wj/29+YUUUVqcJyXxM/5J7qn/bL/ANGpXyb428Q6p4m8W6hqGrrPFcea0a2sx5tUVjiHGBjbyDwMnJPJNfXHxDtbi98C6lb2lvLPO/lbY4kLs2JUJwBz0BNfP3jzwBquueMb/V9B0HVobS9fz3iurKRWSVvv4xuyC2W6jG4jGAM4OajUd+y/U9SGHqV8FBU1e0p31XVQ/wAj1z4K+F9A0zwRp2t6dHHPqF/b4ub0oysSHbdGAxO0Kw2krgNsDelfNnjnW/8AhI/HOtastx9ohnu38iXZs3QqdsfGBj5Ao5GfXnNaf/Cq/GH/AEBrv/wFm/8AiK0NT+EusReV/ZNrrV1nPmfa9Je329MY2s+e/XGMDrnivbQ7mP8AZuK/l/Ff5nUeFvFnh23+Cr+Gf7Ynl1u5lMv2OVJGWLEwO1G27VXYm/GfvM3c4r6Ur4/0v4beLLHUoriXRb4omchbWXPII/u+9fYFKElKo2uy/U1xVGdHCU4VFZ8038rQ/wAgooorY8w5L4mf8k91T/tl/wCjUrM+Nv8AySHXf+3f/wBKI60/iZ/yT3VP+2X/AKNSvEfGnwo+KGp6yxuLqTxFAHeSGc3ioqbmOQI5GAjJAUlUyo4AJxxkv4r9F+p3VP8Acaf+Of5Uzm/gl/yV7Qv+3j/0nkr6v13TP7b8PanpPneT9utJbbzdu7ZvQruxkZxnOMivlD/hSXxD/wChe/8AJ23/APjlbF98PvjPqdnJZ341W7tZMb4Z9ZjkRsEEZUy4OCAfwrU4Th/An/JQ/DX/AGFbX/0atex/tJ6Apt9G8RxrGHVzYzsWbcwILxgDpgYlyeD8w69vMNS+EvjzSrdZ7jw1dujOEAtWS4bOCeVjZiBx1xjp6ivovw34Z13xB8L5PD3xDG+4mzFmKYGYRqQY2dwSpkDLnPIIC7sktkA8o/Zz8Q2+n+KNR0OdcSapErwPk8vEHJTAHdWY5JGNmOSRX0vXyJ4r+C/i/wANXBNvYyaxZs+2OewjaRurY3Rj5lOFyeqjIG4msP8A4Szx5/aP9nf8JB4j+3eb5H2b7bP5nmZ27Nuc7s8Y65oA9r+KHxhvvCPjyw0vSFjmgskD6lA4wJy4BWPJXKlVwwZSRlxkHaQfMPjjBND8W9XeWKREmSB4mZSA6+Si5X1G5WGR3BHau4+Bvw0mj1S717xLol3by2boNPS8jMY8zks+xhklfk2k8ZJxkrlfS/iJ8MtL+INnEZZPsWpwYEN8ke8hM5KMuRuXkkcjB5B5YEA5v9nrX11LwHLo7NH5+lXDKEVWB8qQl1ZieCS3mjjsoyO59Q1XVbHQ9LuNT1O5jtrO3TfLK/RR/MknAAHJJAGSa+RLz4efEDwjrg+y6TqouotxhvdKWSQFTlcq8fK5GeDhsHkDNR22l/Ebx5b20Kp4g1azlcvC91LI1uWUMCQ8h2Aj5hnPXI6nFAHqfgb4pa74w+NEsNms/wDwj11Ey/ZJUDfZ440JWTK/dZnwCSSP3gXJwhHkHxHsbjT/AIleI4bqPy5G1CaYDcDlJGMiHj1VlPtnnmvpP4TfDmHwP4fS4vbaMeILtP8AS5N4fy1zkRKcYAAwWxnLdyAuLHxM+Gdj4/0sOhjttat0Itbsjgjr5cmOShPfqpOR1IYAk+EfiG38Q/DXSXgXZJYRLYTpkna8SgDkgZyu1uM43YySDXUa3reneHNHn1bVrj7PYwbfMl2M+3cwUcKCTyQOBXyYfAPxL8IXCXFppOs2s86MnmaXIZG2ggkMYGJAzg4OM49quWPwz+JPji8judSt75duYftetzupjABYDD5kK5PG1SMk++AD2v4bfF+H4g6zd6W2jSafPDb/AGhGFwJldQwVgflUg5ZccHPPTHJ8c/C7eIfh9Nd28cZu9Jf7YCUXcYgCJVDEjaNvznrnywME4x0ngXwLpfgPQxYWA824kw11duuHuHHc+ijJwvbPckk9RQB8MeG/FOs+EdRkv9DvPsl1JEYWfykkyhIJGHBHVR+VemfADwZfah4nTxYZI4tP015IRnlppWjKlQM8ALIGJPsADyV87v8Aw3/xcO58L6dJ/wAxV9Pt5Lhv+mpjUuQPpkgfhX2X4c0Cx8LeH7PRdNWQWlqhVPMbczEkszE+pYk8YHPAA4oAxvEH/JQvB3/b7/6KFdbXJeIP+SheDv8At9/9FCutrKn8UvX9Ed2L/g0P8D/9LmFFFFanCcl4g/5KF4O/7ff/AEUK62uS8Qf8lC8Hf9vv/ooV1tZU/il6/ojuxf8ABof4H/6XM+HPGOgN4W8Y6rorLIEtbhli8xlZmiPzRsSvGShU9uvQdKj0211jxhrml6RHcT3d1Jss7bzneQQxjoO5WNRknAwACcV7n+0b4XWfS9P8UQRyGe2cWdztRmHlNlkZjnChWyOnJlHPAFZH7N+gXx1nU/EZWMaetu1iGLfM0paNyAPQKBknH3hjPONThPb/AAf4ah8H+E7DQYLiS4S0RgZnABdmYuxwOg3McDnAxyetfLnxp0T+xPihqmy38m3vtt7F8+7fvHzt1JGZBJwcewxivr+vL/jL8N5fGujwahpMO/W7H5I412L9ojZhlWZiMbeWGTj7wxlsgA4D9m7WbO01zWtImfZdX8UUtvkgB/K37lGTkth8gAHhWPGK+j6+EP8AiceGdY/5ftK1O3/34Joty/gwyrfiD71oX3jnxTqehyaLf69fXdhJKJnjnlMhZhjALH5iowDtztyM4zzQB7vqPxxef4lad4b0C2sbnTZNQhtJr9maTzg7IrGMDaF2kuM/MGwCOOvMftKan5viHQ9J8nH2a0e583d97zX27cY4x5Oc553dsc5fw0+F+v6l4fm8Z6dqMlhfQo76MsJUtPKhIO/JwEbDR4PXJJ+UYfz/AMSweKluILjxTFrIndCkMmqLLuZVOSFMnJALZwOm73oA9E/Zx/5KHqH/AGCpP/RsVev/ABt/5JDrv/bv/wClEdfJmlarfaHqlvqemXMlteW774pU6qf5EEZBB4IJByDXUeLfHvjzVbeXQfE9/doiurzWctqlu2cbl3hUUkchgDx0PYGgDi6+/wCvgCvpf4ca54x8f/CrWLEX/wBk1C3xaWWss+XlOAzK+PmDBSq+YOfnBGWUkgEk/wC0b4Zh1mW3TTNSn09EIW7jChncNjiNiPkK87iQexUda19N8Y6X448TeD9W0nz1hWW/hkjnTa8biFTg4JB4Kngnr65A+Vb+xuNM1G5sLyPy7q1leGZNwO11JDDI4OCD0r0v4C/8lCt/+Bf+ipayrfD81+aO7Lv4z/wVP/SJH1ZRRRWpwhRRRQAUUUUAFFFMlljgheaaRY4o1LO7nCqBySSegoBK+iH0Vk/8JT4e/wCg9pf/AIGR/wCNH/CU+Hv+g9pf/gZH/jU88e5v9Vr/AMj+5mtRWT/wlPh7/oPaX/4GR/40f8JT4e/6D2l/+Bkf+NHPHuH1Wv8AyP7ma1FZP/CU+Hv+g9pf/gZH/jR/wlPh7/oPaX/4GR/40c8e4fVa/wDI/uZrUVk/8JT4e/6D2l/+Bkf+NH/CU+Hv+g9pf/gZH/jRzx7h9Vr/AMj+5nL6zrum+H/ikl3qlz9ngfRRGrbGfLGcnGFBPQGtH/hZnhD/AKC//ktL/wDEVr/8JT4e/wCg9pf/AIGR/wCNH/CU+Hv+g9pf/gZH/jWKum+WS1/ruejJQqRiqtCbaSWjstPJwf5mR/wszwh/0F//ACWl/wDiKP8AhZnhD/oL/wDktL/8RWv/AMJT4e/6D2l/+Bkf+NH/AAlPh7/oPaX/AOBkf+NPml/Mvu/4JHsaH/Pip/4Ev/lZkf8ACzPCH/QX/wDJaX/4ij/hZnhD/oL/APktL/8AEVr/APCU+Hv+g9pf/gZH/jR/wlPh7/oPaX/4GR/40c0v5l93/BD2ND/nxU/8CX/ysyP+FmeEP+gv/wCS0v8A8RR/wszwh/0F/wDyWl/+Irfs9a0rUZjDY6nZ3UqruKQTq7AdM4B6cj86vU17R7SX3f8ABMpvBwdpUpp/41/8rOS/4WZ4Q/6C/wD5LS//ABFH/CzPCH/QX/8AJaX/AOIrrazrrX9Gsrh7e71ewgnTG6OW5RGXIyMgnPQg0P2i3kvu/wCCOH1SbtClN+k1/wDKz5g+Lmoal4w8bzz2VxJeaPAiJYgkIqAopfCtg5L5ySMnA7AVc+EOk6DouuPrviu5gjktuLK0eGSUiTg+cSoKjHIAOTk54KqT9Hf8JT4e/wCg9pf/AIGR/wCNH/CU+Hv+g9pf/gZH/jS5pfzL7v8AgmnsaH/Pip/4Ev8A5WZH/CzPCH/QX/8AJaX/AOIo/wCFmeEP+gv/AOS0v/xFa/8AwlPh7/oPaX/4GR/40f8ACU+Hv+g9pf8A4GR/40c0v5l93/BD2ND/AJ8VP/Al/wDKzl9G13TfEHxSe70u5+0QJopjZtjJhhODjDAHoRXfVk/8JT4e/wCg9pf/AIGR/wCNH/CU+Hv+g9pf/gZH/jTptRTvJGeKhUrSi4UpJJJa6vTzsvyNaisn/hKfD3/Qe0v/AMDI/wDGj/hKfD3/AEHtL/8AAyP/ABrTnj3OX6rX/kf3M1qKyf8AhKfD3/Qe0v8A8DI/8aP+Ep8Pf9B7S/8AwMj/AMaOePcPqtf+R/czWorJ/wCEp8Pf9B7S/wDwMj/xo/4Snw9/0HtL/wDAyP8Axo549w+q1/5H9zNaisn/AISnw9/0HtL/APAyP/Gj/hKfD3/Qe0v/AMDI/wDGjnj3D6rX/kf3M1qKyf8AhKfD3/Qe0v8A8DI/8aP+Ep8Pf9B7S/8AwMj/AMaOePcPqtf+R/czWorJ/wCEp8Pf9B7S/wDwMj/xo/4Snw9/0HtL/wDAyP8Axo549w+q1/5H9zNaiiiqMAooooAKKKKACiiigAooooAKKKKAOS/5q9/3Af8A2vXW1zus+ELfWdXTVP7T1SxulgFvusZxFlNxbB+UnqfXsKqf8IJ/1Nfij/wY/wD2NYLni3p1PTqfVq0YN1LNRStZvY62iuS/4QT/AKmvxR/4Mf8A7Gj/AIQT/qa/FH/gx/8Asarnn/L+Jl7DC/8AP7/yVnlnjb4/axo/i3UNK0TTLH7LYytbO96js8kiMVZhtcALkYA5PGeM4FT4a2GtfFbxrL4r8SX12bPSbhJ7SOJisST7lYRxqQVCBUG4AhuUJJJJr0Kf4I+Fbq4luLiW/mnlcvJJI0TM7E5JJMeSSec1qWPw2s9Ms47Ow8QeIbS1jzshgvBGi5JJwoXAyST+NHPP+X8Q9hhf+f3/AJKztaK5L/hBP+pr8Uf+DH/7Gj/hBP8Aqa/FH/gx/wDsaOef8v4h7DC/8/v/ACVnW0VyX/CCf9TX4o/8GP8A9jR/wgn/AFNfij/wY/8A2NHPP+X8Q9hhf+f3/krOtrJ8U/8AIoa1/wBeE/8A6Lasj/hBP+pr8Uf+DH/7GmS/D+OeF4ZvE/iWSKRSro9/lWB4IIK8ilKU2muX8TSjTwtOpGbq7NP4WbXhb/kUNF/68IP/AEWta1V9Ps49O021sYWZoraFIULnLEKABnHfirFaRVopHFXmp1ZSWzbCiiiqMj47+Klhef8ACZatqP2Sf7D9umg+0+WfL8zzHbZu6bsc464r6Y8JfEfwz4zt4m07UI4ruR2QWF06pcZUZOEydw285XIxnuCBj3PhHxI9vrumCHw9eaTqt5NcvFemYthyMD5QMEYUgjkEZB6V5v8A8M46n/0ErT/v+3/xquenNxjZpnr4vDxrVnUjUjZ26+SPoyvA/jN8TLHWbA+CvDYj1WW9eIT3FufNXO9WSOLb99ywXJGQOnJJ25//AAzjqf8A0ErT/v8At/8AGq6PwZ8I9d8Eao+p2A0K6vCmyOW9eVzCDnds2qoBIOCeTjgYBOb9r5M5vqP/AE8h953Pw18Ef8IF4SXSpLr7TdSym5uXUYQSMqqVTjO0BQMnk8njOB2Fcl/xcP8A6lf/AMmKP+Lh/wDUr/8AkxR7XyYfUf8Ap5D7zraK5L/i4f8A1K//AJMUf8XD/wCpX/8AJij2vkw+o/8ATyH3nW0VyX/Fw/8AqV//ACYo/wCLh/8AUr/+TFHtfJh9R/6eQ+862iuS/wCLh/8AUr/+TFH/ABcP/qV//Jij2vkw+o/9PIfedbRXJf8AFw/+pX/8mKP+Lh/9Sv8A+TFHtfJh9R/6eQ+862iuS/4uH/1K/wD5MUf8XD/6lf8A8mKPa+TD6j/08h951tFcl/xcP/qV/wDyYo/4uH/1K/8A5MUe18mH1H/p5D7zraK5L/i4f/Ur/wDkxR/xcP8A6lf/AMmKPa+TD6j/ANPIfedbRXJf8XD/AOpX/wDJij/i4f8A1K//AJMUe18mH1H/AKeQ+862iuS/4uH/ANSv/wCTFH/Fw/8AqV//ACYo9r5MPqP/AE8h951tFcl/xcP/AKlf/wAmKP8Ai4f/AFK//kxR7XyYfUf+nkPvOtorkv8Ai4f/AFK//kxR/wAXD/6lf/yYo9r5MPqP/TyH3nW0VyX/ABcP/qV//Jij/i4f/Ur/APkxR7XyYfUf+nkPvOtrkvD/APyULxj/ANuX/oo0f8XD/wCpX/8AJipvDGjazZazrOqa09gZ9Q8jC2RfavlqV/iGehHc96lycpRsno/0ZtClGhRrc04vmikknd354v8AJM6eiiitzywrkvHf/Mtf9h61/wDZq62uN+It1DZWegXdw+yCDWreSRsE7VUOScDnoKyr/wANnflibxcEvP8AJnZUVyX/AAszwh/0F/8AyWl/+Io/4WZ4Q/6C/wD5LS//ABFHt6X8y+8X9l47/nzP/wABf+R1tFcl/wALM8If9Bf/AMlpf/iKP+FmeEP+gv8A+S0v/wARR7el/MvvD+y8d/z5n/4C/wDI62iuS/4WZ4Q/6C//AJLS/wDxFH/CzPCH/QX/APJaX/4ij29L+ZfeH9l47/nzP/wF/wCR1tFcl/wszwh/0F//ACWl/wDiKP8AhZnhD/oL/wDktL/8RR7el/MvvD+y8d/z5n/4C/8AI62iuS/4WZ4Q/wCgv/5LS/8AxFH/AAszwh/0F/8AyWl/+Io9vS/mX3h/ZeO/58z/APAX/kdbRXJf8LM8If8AQX/8lpf/AIij/hZnhD/oL/8AktL/APEUe3pfzL7w/svHf8+Z/wDgL/yOtorkv+FmeEP+gv8A+S0v/wARR/wszwh/0F//ACWl/wDiKPb0v5l94f2Xjv8AnzP/AMBf+R1tFcl/wszwh/0F/wDyWl/+Io/4WZ4Q/wCgv/5LS/8AxFHt6X8y+8P7Lx3/AD5n/wCAv/I62iuS/wCFmeEP+gv/AOS0v/xFH/CzPCH/AEF//JaX/wCIo9vS/mX3h/ZeO/58z/8AAX/keWftKa3/AMgPQIrj+/e3EGz/AIBE27H/AF2GAfqOlemfCzStFsPh9otzpFtaI93ZQvdTw4LSy4y+9+pKuXGCfl5AxjFed/F8+EvHWlw32maxGmtWKMIla1kUXKHnyy2zIIOSpJwCWBxu3Dg/B3jjx74Js/sNgkFzYDcUtLza6RsxBJUhgw6HjO35mOMnNHt6X8y+8P7Lx3/Pmf8A4C/8j6svrCz1Ozks7+0gu7WTG+GeMSI2CCMqeDggH8K8j/aCg0G1+H1hbyRRw3kVwiaZHAqLsUDDjHURBcAhf4vLzxXL/wDC7viJ/wBAPQ/++H/+PVw9jpt/4x8Wx3XjfXJ4LU5ae6kJlfbuJ8uJVBC5LHAwFXk4PCk9vS/mX3h/ZeO/58z/APAX/keyfs4/8k81D/sKyf8AoqKvYK8+8OeLfAPhbw/Z6LpuqyC0tUKp5kEzMxJLMxOzqWJPGBzwAOK1P+FmeEP+gv8A+S0v/wARR7el/MvvD+y8d/z5n/4C/wDI62iuS/4WZ4Q/6C//AJLS/wDxFH/CzPCH/QX/APJaX/4ij29L+ZfeH9l47/nzP/wF/wCQeO/+Za/7D1r/AOzV1tebeIvGGg+ILzw7aaXffaJ01q2kZfJdMKCRnLKB1Ir0mlTkpTk4u+xrjaNSjh6MKsXF+9o1Z7+YUUUVseYFFFFABRRRQAUUUUAFFFFAHJfEz/knuqf9sv8A0aldbXJfEz/knuqf9sv/AEaldbWS/iv0X6ndU/3Gn/jn+VMKKKK1OEKKKKACiiigAooooAKKKKACiiigAooooAK4/wCJ3i7/AIQvwNe6lE22+l/0ay4z++cHDfdI+UBnwRg7cd64P9oXxJrWh2/h+DSdUu7BLl7h5jaymNnKCMLllwcfO3GcHjPQY8I8L+Eda8Y6olho9lJMS6rLOVIigByd0j4wowre5xgAnigD0D4GfD+bxB4gh8TXfljS9LuMqpY7prhQGUDBBAUlGJPB4GDlsfUdZfhzQLHwt4fs9F01ZBaWqFU8xtzMSSzMT6liTxgc8ADitSgDkvEH/JQvB3/b7/6KFdbXJeIP+SheDv8At9/9FCutrKn8UvX9Ed2L/g0P8D/9LmFFFFanCcl4g/5KF4O/7ff/AEUK62sPX/C9v4guLK4kvr+zns/M8qSylEbDeADzgnoMcY6ms7/hBP8Aqa/FH/gx/wDsaxXPGUrK9/8AJHpyeHrUaSlU5XFNPRv7Un+TOtorkv8AhBP+pr8Uf+DH/wCxo/4QT/qa/FH/AIMf/safPP8Al/Ey9hhf+f3/AJKzraK5L/hBP+pr8Uf+DH/7Gj/hBP8Aqa/FH/gx/wDsaOef8v4h7DC/8/v/ACVnQ6rpVjrml3GmanbR3NncJslifow/mCDggjkEAjBFZf8Awgng/wD6FTQ//BdD/wDE1S/4QT/qa/FH/gx/+xo/4QT/AKmvxR/4Mf8A7Gjnn/L+Iewwv/P7/wAlZ1tFcl/wgn/U1+KP/Bj/APY0f8IJ/wBTX4o/8GP/ANjRzz/l/EPYYX/n9/5Kzrar31hZ6nZyWd/aQXdrJjfDPGJEbBBGVPBwQD+Fc1/wgn/U1+KP/Bj/APY0f8IJ/wBTX4o/8GP/ANjRzz/l/EPYYX/n9/5Ky7/wgng//oVND/8ABdD/APE1sX81xb6dczWdr9ruo4neG38wR+a4BKpuPC5OBk9M1zX/AAgn/U1+KP8AwY//AGNH/CCf9TX4o/8ABj/9jRzz/l/EPYYX/n9/5Kz5B8R6/feKfEF5rWpNGbu6cM/lrtVQAFVQPQKAOcnjkk816X8CdNmi8U6dqjNH5FxcXFuigncGjg3MTxjGJVxz2PTv6n/worwf/wBPf/kH/wCN101h4Jt7LWbTVJNY1m+ntN/lLe3QlVdylT/Dnoex7CplzySXL1XXzNqH1ag5TVS/uyVuV7uLS/M6eiiitzywooooAKKKKACsnxT/AMihrX/XhP8A+i2rWrJ8U/8AIoa1/wBeE/8A6Lapn8LN8L/Hh6r8zL8N+G9Cn8LaRNNounSSyWULO72qFmJQEkkjk1qf8It4e/6AOl/+Acf+FHhb/kUNF/68IP8A0Wta1RCEeVaG+JxNZVppTe76vueO/G+XTPCvgqJdK07SLa/v7j7OGFrGJVi2sXaPjIIOwbu2/jBII4P4E61a3viibw9rdvZXsN3E8tqbu1EsgmUAlQ5BwpQOcHjKjGCTuxvjnrP9r/FC9iV4JIdPijs43hOc4G9gxyfmDu6npjbjGQaw/hjqU2lfE3w7cQLGzvepbkOCRtlPlMeCOdrkj3x16VfJHsYfWq/87+9n17/wi3h7/oA6X/4Bx/4Uf8It4e/6AOl/+Acf+Fa1FHJHsH1qv/O/vZk/8It4e/6AOl/+Acf+FH/CLeHv+gDpf/gHH/hWtRRyR7B9ar/zv72ZP/CLeHv+gDpf/gHH/hR/wi3h7/oA6X/4Bx/4VrUUckewfWq/87+9mT/wi3h7/oA6X/4Bx/4Uf8It4e/6AOl/+Acf+Fa1FHJHsH1qv/O/vZ4j8TfhVr+veIftnhSKysbG309AYUl8j7RNvkJCqoxuxsGW2jkc8HHz/fyavpmo3NheXNxHdWsrwzJ55O11JDDIODgg9K+7q+IPHf8AyUPxL/2Fbr/0a1HJHsH1qv8Azv72fR3w/t4INd0BoYY42n8IW80pRQDJIzLlmx1Y9yea9RrzLwJ/yG/DX/YmWv8A6Etem1FJWvbudGYScnTcnd8qCuH03StO1P4heLft9ha3fl/Y9nnwrJtzEc4yOM4H5V3Fcl4f/wCSheMf+3L/ANFGiqk5RT7/AKMeClKFKvKLs+Rf+lwNf/hFvD3/AEAdL/8AAOP/AAo/4Rbw9/0AdL/8A4/8K1qKvkj2Ob61X/nf3syf+EW8Pf8AQB0v/wAA4/8ACj/hFvD3/QB0v/wDj/wrWoo5I9g+tV/5397Mn/hFvD3/AEAdL/8AAOP/AAri/ip4KN54DvP+EX0y2g1CF0lKWlnGJZ4wfmRWGCp/i+Xk7duDur0qijkj2D61X/nf3s+DZ7rVLW4lt7ie8hnicpJHI7KyMDggg8gg8Yr6Y+CegQah8N7a+1uy0+/ee4la2klt0eRYg23a7Fck71cjJPBAzxgeRfHWxuLT4r6lNPHsju4oJoDuB3oI1jJ46fMjDn09MV738GIJrb4SaCk8UkTlJXCupUlWmdlPPYqQQe4INHJHsH1qv/O/vZ03/CLeHv8AoA6X/wCAcf8AhR/wi3h7/oA6X/4Bx/4VrUUckewfWq/87+9mT/wi3h7/AKAOl/8AgHH/AIUf8It4e/6AOl/+Acf+Fa1FHJHsH1qv/O/vZk/8It4e/wCgDpf/AIBx/wCFH/CLeHv+gDpf/gHH/hWtRRyR7B9ar/zv72ZP/CLeHv8AoA6X/wCAcf8AhR/wi3h7/oA6X/4Bx/4VrUUckewfWq/87+9mT/wi3h7/AKAOl/8AgHH/AIVy/wAQ9A0ay8C6lcWmkWEE6eVtkitkRlzKgOCBnoSK76uS+Jn/ACT3VP8Atl/6NSsq0I+zlp0Z25biazxtFOb+KPV90dbRRRW55YV5t4P8O3HiDwtZapd+J/EaTz79yxX5Cja7KMZBPQDvXpNcN4J1KHRvhJDqlwsjQWVvc3EixgFiqPIxAyQM4HqKxnFSqJPs/wBD08LWnRwlSdN2fNBfK0/8i5/wgn/U1+KP/Bj/APY0f8IJ/wBTX4o/8GP/ANjXAeG/2hYdc8S6bpM/hqS1S9uEtxMl6JSjOdqnaUXI3EZ54GTz0PtlP2MOxl/aWK/m/Bf5Hxn/AMLU8Yf9Bm7/APAqb/4uvUPg/e6x8QP7Z/tbxFrUP2HyPL+yXrrnf5mc7t39wdMd6P2lrKx+z6BfmWNNQLywiMQ/NNFhSSX7BGIwp/56kjoasfs06bNFo2v6ozR+RcXEVuigncGjVmYnjGMSrjnsenc9jDsH9pYr+b8F/kej/wDCCf8AU1+KP/Bj/wDY0f8ACCf9TX4o/wDBj/8AY11tFHsYdg/tLFfzfgv8jkv+EE/6mvxR/wCDH/7Gj/hBP+pr8Uf+DH/7Gutoo9jDsH9pYr+b8F/kcl/wgn/U1+KP/Bj/APY0f8IJ/wBTX4o/8GP/ANjXW0Uexh2D+0sV/N+C/wAjkv8AhBP+pr8Uf+DH/wCxo/4QT/qa/FH/AIMf/sa62ij2MOwf2liv5vwX+RyX/CCf9TX4o/8ABj/9jR/wgn/U1+KP/Bj/APY11tFHsYdg/tLFfzfgv8jh9Gs7jRviM+l/2vql9atpJuNt9cmXD+cFyOg6D07mu4rkv+avf9wH/wBr11tFFWTS7lZhJzlCT3cUFebfEb4g6x4Ov3Wwsba4tIbNLqdpFJZA0pjz99cjcUGBk8+nT0muG1XSrHXPiTcaZqdtHc2dx4f2SxP0YfaPzBBwQRyCARgiireySfUnAKPNOUop2i3rtc8s/wCGjtT/AOgbaf8Afhv/AI7Xo/hbxJ438XeHLTXLCPw9Ha3W/Yk6zBxtdkOQCR1U96+efit4ctPC3xD1DTdOsZLTTwkT2yMXYMpjXcVZiSw37xnJ5BHavov4Jf8AJIdC/wC3j/0oko9l5sPr3/TuH3Gn/wAXD/6lf/yYo/4uH/1K/wD5MV1tFHsvNh9e/wCncPuOS/4uH/1K/wD5MUf8XD/6lf8A8mK62ij2Xmw+vf8ATuH3HJf8XD/6lf8A8mKP+Lh/9Sv/AOTFdbRR7LzYfXv+ncPuOS/4uH/1K/8A5MUf8XD/AOpX/wDJiutoo9l5sPr3/TuH3HJf8XD/AOpX/wDJij/i4f8A1K//AJMV1tFHsvNh9e/6dw+45L/i4f8A1K//AJMUf8XD/wCpX/8AJiutoo9l5sPr3/TuH3HJf8XD/wCpX/8AJij/AIuH/wBSv/5MV1tFHsvNh9e/6dw+45L/AIuH/wBSv/5MUf8AFw/+pX/8mK62sfXvFWheGPsf9tanBZfbJfJg80n5m7njooyMscKMjJGRR7LzYfXv+ncPuMr/AIuH/wBSv/5MVb8IazqOs2eo/wBqJardWd/JZt9lDBDsC8jcSepPp24rasb+z1OzjvLC7gu7WTOyaCQSI2CQcMODggj8K5rwJ/zMv/Yeuv8A2Wps4zSuzX2ka2GqNwimrWsrbs62iiitzzDmL/xtb2Ws3elx6PrN9PabPNaytRKq7lDD+LPQ9x2NQ/8ACd/9Sp4o/wDBd/8AZUeH/wDkoXjH/ty/9FGtX/hLPDf9o/2d/wAJBpX27zfI+zfbY/M8zO3ZtzndnjHXNYR55XfN1fTzPUr/AFahKMHTv7sXfmfWKb/Myv8AhO/+pU8Uf+C7/wCyrDn+N3hW1uJbe4iv4Z4nKSRyLErIwOCCDJkEHjFek188fG74YaXpVnd+MNPvvs81xd7rm0uJMiV5DyYe+7O5ipJ43EbQuDXJP+b8DH2+F/58/wDkzPRtN+L+g6zcNb6Xp+rX06oXaO1hSVguQMkK5OMkDPuK1P8AhO/+pU8Uf+C7/wCyr59+AOmfb/ihBc+d5f8AZ9pNc7dufMyBFtznj/W5zz93HfI+r6OSf834B7fC/wDPn/yZnJf8J3/1Knij/wAF3/2VH/Cd/wDUqeKP/Bd/9lXW0Uck/wCb8A9vhf8Anz/5Mzkv+E7/AOpU8Uf+C7/7Kj/hO/8AqVPFH/gu/wDsq62ijkn/ADfgHt8L/wA+f/Jmcl/wnf8A1Knij/wXf/ZVo6B4ot/EFxe28djf2c9n5fmx3sQjYbwSOMk9BnnHUVuVyXh//koXjH/ty/8ARRpPnjKN3e/+TNYrD1qNVxp8rik1q39qK/U62iiitjzDD1nxhoPh+8S01S++zzvGJFXyXfKkkZyqkdQazv8AhZnhD/oL/wDktL/8RR/zV7/uA/8AteutrFOpJuzX3f8ABPTnDCUYwU4SbaT0kktfLkf5nJf8LM8If9Bf/wAlpf8A4ij/AIWZ4Q/6C/8A5LS//EV1tFO1Xuvu/wCCZe0wP/Puf/ga/wDlZyX/AAszwh/0F/8AyWl/+Io/4WZ4Q/6C/wD5LS//ABFdbRRar3X3f8EPaYH/AJ9z/wDA1/8AKzkv+FmeEP8AoL/+S0v/AMRR/wALM8If9Bf/AMlpf/iK62ii1Xuvu/4Ie0wP/Puf/ga/+VnJf8LM8If9Bf8A8lpf/iKP+FmeEP8AoL/+S0v/AMRXW18ifEz4X614JcareajHqlneXBT7YSVlaVl3HzFYk5JD8hmztycEgUWq9193/BD2mB/59z/8DX/ys+jv+FmeEP8AoL/+S0v/AMRR/wALM8If9Bf/AMlpf/iK8Z/Zs02aXxVrOqK0fkW9kLd1JO4tI4ZSOMYxE2ee469vpOi1Xuvu/wCCHtMD/wA+5/8Aga/+VnJf8LM8If8AQX/8lpf/AIij/hZnhD/oL/8AktL/APEV1tFFqvdfd/wQ9pgf+fc//A1/8rMPRvGGg+ILx7TS777ROkZkZfJdMKCBnLKB1Ircrkv+avf9wH/2vXW06cm0+boRjKVOnKLpXSaT1d3r5pL8grO1nXdN8P2aXeqXP2eB5BGrbGfLEE4woJ6A1o1yXjv/AJlr/sPWv/s1OpJxi2hYKjCtXjTns+2+3zD/AIWZ4Q/6C/8A5LS//EUf8LM8If8AQX/8lpf/AIiutoqbVe6+7/gmntMD/wA+5/8Aga/+VnJf8LM8If8AQX/8lpf/AIij/hZnhD/oL/8AktL/APEV1tFFqvdfd/wQ9pgf+fc//A1/8rOS/wCFmeEP+gv/AOS0v/xFH/CzPCH/AEF//JaX/wCIrraKLVe6+7/gh7TA/wDPuf8A4Gv/AJWcl/wszwh/0F//ACWl/wDiKP8AhZnhD/oL/wDktL/8RWz4k0hvEHhrUtHS8ksze27weeiK5UMMHhuCCOD0OCcEHBHx5438Bax4B1G2s9WMEv2mLzY5rYu0ZwSCu5lX5hwSB0DL60Wq9193/BD2mB/59z/8DX/ys+qv+FmeEP8AoL/+S0v/AMRR/wALM8If9Bf/AMlpf/iK4z9nH/knmof9hWT/ANFRV7BRar3X3f8ABD2mB/59z/8AA1/8rOYtfiH4WvbyC0t9U3zzyLHGv2eUbmY4AyVx1NdPXJeO/wDmWv8AsPWv/s1dbThKV2pdBYqlRVOFSimua+jaez8kgooorQ4gorLl8SaFBM8M2tadHLGxV0e6QMpHBBBPBpv/AAlPh7/oPaX/AOBkf+NTzx7m6w1Z6qD+5mtRWT/wlPh7/oPaX/4GR/40f8JT4e/6D2l/+Bkf+NHPHuH1Wv8AyP7ma1FZP/CU+Hv+g9pf/gZH/jR/wlPh7/oPaX/4GR/40c8e4fVa/wDI/uZrUVnWuv6Ne3CW9pq9hPO+dscVyjs2Bk4AOegJrRppp7Gc6c4O01b1Cobq6t7K3e4u7iKCBMbpJXCKuTgZJ46kCpq5L4mf8k91T/tl/wCjUqakuWDl2NcJRVfEU6LduZpfe7Gv/wAJT4e/6D2l/wDgZH/jR/wlPh7/AKD2l/8AgZH/AI0f8It4e/6AOl/+Acf+FH/CLeHv+gDpf/gHH/hU/vfI2/2H+/8AgU9T1bwlrOnS2F/rGlzWsuN6fbkXOCCOQwPUCue/4Rz4Xf8APfS//Bq3/wAcq7448FxX3gvVLbw1pGlwavJFi3cW0aHqNwVivysV3AHjBIOR1Hynr+neJfC2qNputC8tLsIH2NNuDKehVlJDDqMgnkEdQaiVOUneSTOmjjKVCPLSqVIryaX5H0//AMI58Lv+e+l/+DVv/jlH/COfC7/nvpf/AINW/wDjleQfAKKPWfHV9b6oi30C6ZI6x3Q81Q3mxDIDZGcEjPua+jP+EW8Pf9AHS/8AwDj/AMKn2H92P3Gv9qf9Pqv/AIF/wTkv+Ec+F3/PfS//AAat/wDHKP8AhHPhd/z30v8A8Grf/HK63/hFvD3/AEAdL/8AAOP/AAo/4Rbw9/0AdL/8A4/8KPYf3Y/cH9qf9Pqv/gX/AATkv+Ec+F3/AD30v/wat/8AHKP+Ec+F3/PfS/8Awat/8crrf+EW8Pf9AHS//AOP/Cj/AIRbw9/0AdL/APAOP/Cj2H92P3B/an/T6r/4F/wTkv8AhHPhd/z30v8A8Grf/HKP+Ec+F3/PfS//AAat/wDHK63/AIRbw9/0AdL/APAOP/Cj/hFvD3/QB0v/AMA4/wDCj2H92P3B/an/AE+q/wDgX/BOS/4Rz4Xf899L/wDBq3/xyj/hHPhd/wA99L/8Grf/AByut/4Rbw9/0AdL/wDAOP8Awo/4Rbw9/wBAHS//AADj/wAKPYf3Y/cH9qf9Pqv/AIF/wTkv+Ec+F3/PfS//AAat/wDHKP8AhHPhd/z30v8A8Grf/HK63/hFvD3/AEAdL/8AAOP/AAo/4Rbw9/0AdL/8A4/8KPYf3Y/cH9qf9Pqv/gX/AATkv+Ec+F3/AD30v/wat/8AHKP+Ec+F3/PfS/8Awat/8crrf+EW8Pf9AHS//AOP/Cj/AIRbw9/0AdL/APAOP/Cj2H92P3B/an/T6r/4F/wTkv8AhHPhd/z30v8A8Grf/HKP+Ec+F3/PfS//AAat/wDHK63/AIRbw9/0AdL/APAOP/Cj/hFvD3/QB0v/AMA4/wDCj2H92P3B/an/AE+q/wDgX/BOS/4Rz4Xf899L/wDBq3/xyj/hHPhd/wA99L/8Grf/ABytnxL4P0y58K6vBpehaeNQkspktTHbxowlKEJhsDad2OcjFfKfiTwz4z8I+WdctdQtI5MBZvO8yMk5wu9CV3fKTtznAzjFHsP7sfuD+1P+n1X/AMC/4J9PaZp/w60bUYr+wvdLhuos7H/tLdjIIPBcjoTXW2Oq6dqfmfYL+1u/Lxv8iZZNuc4zg8Zwfyr4k0KbV9Q8Q6ZZWd8RdXF3FFCbhi8YdnAXepBBXJGQQcjsa+tPC1rb2Xjrxdb2lvFBAn2PbHEgRVzExOAOOpJq43ptRskn29P+AY1nTxcJ1XOcpQSfvO+nMlb/AMmudlRRRW55IUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABWT4p/wCRQ1r/AK8J/wD0W1a1ZPin/kUNa/68J/8A0W1TP4Wb4X+PD1X5h4W/5FDRf+vCD/0WtZPxJ8UN4Q8B6lqsEka3mwQ2m51B81ztBUEEMVBL7cHIQ9smtbwt/wAihov/AF4Qf+i1rwb9onxd9s1i08KWzfubHFzd8dZmX5F5X+FGzkEg+Zg8rRD4UGK/jz9X+ZzfwY+H9p438QXU+qeXJpenIpntizq0zSBwgBUggAqWJz2Awckjz+eG+0LWZYHMlrqFhcFGMcnzRSo2OGU9Qw6g9uK6zwR8VfEHgLTrmw0yKxntZ5fOKXcTNsfABIKsp5AXrn7oxjnPJ6tqU2s6zfapcLGs97cSXEixghQzsWIGSTjJ9TVGB9p+B/En/CXeC9L1wx+XJdRfvUC4AkUlH2jJ+XcrYyc4xnmo/H2o61pXgjU7zw7bSXGrIiLbxxwGZss6qWCDqVUluhHHIIyK8z/Z28XfbNHu/Cly376xzc2nHWFm+deF/hds5JJPmYHC17hQB8gf8Lt+If8A0MP/AJJW/wD8brrLz44eN/FVva2HhbRZLbUIU827ksoTdtIAACVQodibj33HlRu655v462NxafFfUpp49kd3FBNAdwO9BGsZPHT5kYc+npivoP4U6BY+H/hzpC2KyZvreO+uGdtxaWSNST6AAYAA7AZyckgHhFt8T/if4KuJJdZiu3S9eSRItas3VS+V3GP7pAHA2qdo3fd5r6L8F+LLPxr4Xtdas08rzcpNAXDtDIpwykj8CM4JUqcDOK8z/aTmsV8K6NBII/7Qe9LwEx5bylQiTDY4G5osjPPHXHB+zZeK/hXWbEXEbPDeiYwiJgyB0ADF84IPlkAAZG0kk7hgA9U8T+J9L8I6HNq+rz+Vbx8Kq8vK56Ig7scH8iSQASPDJ/jx4y8R3Etp4U8Mxq/2clljikvJ4znHmDaAABuXgqRnrnOKx/2hdfbUvHkWjq0nkaVbqpRlUDzZAHZlI5IK+UOe6nA7n0v9n7w9b6Z8P/7YVt91q0rPIcEbUjZo1Trg4Ids4H38c4FAHEWPxg+JPhWKObxX4enurF5SDNeWT2bklTtRXChByC3Kkn5ufTx/XdT/ALb8Q6nq3k+T9uu5bnyt27ZvcttzgZxnGcCvuueCG6t5be4ijmglQpJHIoZXUjBBB4II4xXwprumf2J4h1PSfO877Ddy23m7du/Y5XdjJxnGcZNAH1V4Tmt7jxZoc1na/ZLWTwjbvDb+YZPKQupVNx5bAwMnrivRa5L/AJq9/wBwH/2vXW1lS+16ndjv+Xf+BBXh/j/xf4j8G674rvNB0+ORJXtI7m9dd4tMwkIdnqWPDHKgqAQdwr3CuO0eCG68deNbe4ijmglSzSSORQyuphYEEHggjjFFT4o+v6MMJ/Br/wCBf+lwPnmx+Ovj60vI55tUgvY1zmCe0jCPkEclFVuOvBHT04r6f/tDVLTwb/aV5p3m6vFp/nzWFuc75xHuaJCN3VsqMbvxr408Y6A3hbxjquissgS1uGWLzGVmaI/NGxK8ZKFT269B0r7L8Kaz/wAJD4S0nVy8DyXdpHLL5ByiyFRvUcnGG3DBORjB5rU4Txf/AIaa/wCpR/8AKl/9qr0T4YePdS8eaXPdX+gyackKRhLkMxiu2O4OY8qMBWTpubGcE8c/OnxgvrfUPivr81rJ5kaypCTtIw8caRuOfRlYe+OOK+j/AIR+IbfxD8NdJeBdklhEthOmSdrxKAOSBnK7W4zjdjJINAHN+Ofjivg7xLJo6eGbu5EaA+fcStbCQ5IJQNGSyZGN/QkHGQAT2Hw88cQ+P/DTatFYyWTx3D28sLSBwGAVsq2BkbWXqBzkdsnzv4yaxb+LvEejfDawSdr5tQhlurhIiwtwUI4X+PCSFycgADr12+0WFjb6Zp1tYWcfl2trEkMKbidqKAFGTycADrQB8kfG3/kr2u/9u/8A6Tx13+hftA6PongbTNO/sW+m1OxtIrby96LC+wBN3mcsMqN2Nh54/wBquM+PM19L8VLxLsSCCK3hSz3R7QYtgY7Tj5h5jSc885HbA9n+Cvg+x0PwRp2sPpscOsajb757gvvZ4mdmjxyQoKFCQMZwM8jgA5ux/aMtxrkdhrfhmfS4RKYrmU3Jke3IyDuj8tTweo6jngkYPtFjf2ep2cd5YXcF3ayZ2TQSCRGwSDhhwcEEfhXl/wC0HZ6XL8Olur07L6C7jFiwTJZ2++hODhSgZjyMlF54ANP9nXX21DwdfaLK0jPpdwGjyqhVilywUEck71lJz/eHPYAHsleV+KPj14U0NHi0ppNavFdk2QZjiUqwBzKwwQRkgoGBx2BBrjP2ifGFw2o2nhOzudtqkQuL1YpQfMcn5EcAZG0KHwTzvU44BrT+CXgHwhd+H21a7l03XdSnQebbSIsi2KknCmNhkOSp+YjtheMswBqeE/2gdC1zUWs9as/7D3bRBM85mjdicEMwRdnUHJ+XGckYGfVJ9VsbbRpdYe5jOnx25umuI/nUxBd28bc7ht54zntXjfxt+G3h+38IS+JNKs4NMutP8tXitIVSOdHkC4KjADAvncOoyDngrX/Z8vH17wl4j8MakPP0yLaFQuwOydXEiA5+VfkyNuOWY55oA6jU/j94GsPK+zXF9qW/O77JalfLxjGfNKdc9s9DnHGdDXvjP4K0PTrO8TUv7T+1cpDp+2SRFxnLqWXZ1Aw2GyenBx8+fFnwVpvgPxVa6Xpc93NBLZJcM106swYu64G1VGMIO3rXcfCP4QaB4m8PweItanu7pJHdBYhGgjBUspy/WQfdIKFQCCDnBAAOvg/aK8GzXEUT2mswI7hWlkt4yqAn7x2yE4HXgE+gNdH431Wx1z4U3mp6Zcx3NncJC8UqdGHnJ+IIOQQeQQQcEV5x8dvAXhbRPD0Ov6bZf2dfSXaW/lWiAQy5QnlMgJgITlRyScg53Lzfw0vriT4WeNrBpM2sMtnNGm0fK7yYY568iNPy9zWVf+FL0Z3ZX/v1H/HH80fUlFFFanCFeFy/ELw1ovwcufD9zf79XuNPuYktIY2dgZXkVdxxtXruIJB28gHIz7pXzH4g+HGj/wDCpbnxx9pvv7T+X91vTyf9eIum3d93n73X8qyf8Vej/Q7qf+41P8cPyqHmnhTWf+Ee8W6Tq5edI7S7jll8g4doww3qORnK7hgnBzg8V9l6B4x8OeKUVtF1i0u3KF/JV9sqqG2ktG2HUZxyQOo9RXyZ8NfBH/Ce+LV0qS6+zWsURubl1GXMasqlU4xuJYDJ4HJ5xg/S/g74T+FvBN59usIJ7m/G4Jd3kgd41YAEKAAo6HnG75mGcHFanCeb/tNf8yt/29/+0a2P2fr+z0z4Y6neX93BaWseqvvmnkEaLmOEDLHgZJA/Gq/7SlrZv4e0O8e4230V28UUO8DfG6ZdtvU4KRjI4G7nqK4D4Q/C3/hOLx9V1NtmhWkvlyIj4e5kADeWMcquCCW684XnJUA9v/4Xb8PP+hh/8krj/wCN12mlarY65pdvqemXMdzZ3Cb4pU6MP5gg5BB5BBBwRXm/jL4OeC38Na3e2Gkx2WoC3luIpkumiRJAC4GHby0QkYPAABONuAR5h8AvFzaP4xOh3d7Imn6mhWKJmURi542tyeCygpxyxKDBwMAH1HXN6/4/8KeF3aLWNctIJ1cI0CkyyoSu4bo0BYDHOSMcj1Fc38afGn/CJ+C3trS7nttX1LMVo8K8qqlfMbd/D8pwCPmywI6EjyT4K/DGx8ZPeaxrsckul2r+RHAr7RPKVy24qQwCgqeMZLDngggHo8H7RXg2a4iie01mBHcK0slvGVQE/eO2QnA68An0Br0zRNb07xHo8GraTcfaLGfd5cuxk3bWKnhgCOQRyK4/WPgt4G1ezSAaR9gkjiSGOeykMbqqnOSDlXY8gswZjnrnBHgngzXdS+E/xNex1OWSK0S4+y6lGNwSSPJCygFckLkSKQASOBgMaAPqvU9d0fRPK/tbVbGw87Pl/a7hIt+MZxuIzjI6eoqvY+LPDep3kdnYeINKu7qTOyGC9jkdsAk4UHJwAT+Fef8Axs+Hdv4l0OfxJbzeRqelWju28kpNAm5yhHZhliCOucHqCviHwfsbfUPivoEN1H5kayvMBuIw8cbyIePRlU++OeKAPr++v7PTLOS8v7uC0tY8b5p5BGi5IAyx4GSQPxrH/wCE78H/APQ16H/4MYf/AIqo/GvgrTfHmjQ6Xqk93DBFcLcK1q6qxYKy4O5WGMOe3pXyJ448N/8ACI+NNU0MSeZHay/unLZJjYB03HA+bay5wMZzjigD6y/5q9/3Af8A2vXW15F8K9M/si98PW3nebv8Li53bduPOuTLtxk9N+M98Z46V67WVL7Xqd2O/wCXf+BBXJf81e/7gP8A7Xrra5L/AJq9/wBwH/2vRV6eoYH/AJef4H+h4L+0NY29p8So5oI9kl3p8U053E73DPGDz0+VFHHp65r2f4Jf8kh0L/t4/wDSiSvIP2jv+Sh6f/2Co/8A0bLXr/wS/wCSQ6F/28f+lElanCegVy+s/EbwdoG8aj4isUkSUwvDDJ50iOM5DJHuZcYIORweOteOfG34pXz6pfeD9GuI47BEWK+ni+/I/JeIOGI2YKqwwDlWU8ZB0/DH7OdhLocMvie/votTk+d4bKWMJCD0Qkq25h3IOOcDONxAPZNN8S6DrNw1vpet6bfTqhdo7W6SVguQMkKScZIGfcVqV8wfEn4Rf8K+05PEmh6tfSwxXcSqhixJbcEiQyoRj5wADtXBZRnPX1f4Z+Mbf4meBrqw1f8AfX8URtdTQIY1lSQMFYFT/EoOcYwwbAA20AdhY+LPDep3kdnYeINKu7qTOyGC9jkdsAk4UHJwAT+FakE8N1bxXFvLHNBKgeOSNgyupGQQRwQRzmvizx/4LuPAfih9Hnn+0xmJJoLjYE81GGCdoZtuGDLyf4c9CK93/Z2tbNPA13eW1xfNNLdmK6hmcGFJEGQ0SjplHQMTySvoBQB7BRXD/Fnwtb+KvAN5FcX32L7Bm/SZlLIDGjZ3gAsV2lvujI4ODjafDPgj4Bm8T+JY9dnlkg0/R7iOUFUOZ5gdyoCRjAwC3fBUY+bIAPqusPVPGXhnRbhrbUtf021uEdEeCS5USKXIC5TOQPmByRgDk4HNeb/GT4s3fhO4j8P+HnjTVGRZbi6IST7OpPCBTkbyBk7hwpGAdwK854Z/Z81HVsan4y1We2uJpZHntYSssz5zhmmJZdxbk8Nx3BJwAe76Zruj635v9k6rY3/k48z7JcJLsznGdpOM4PX0NaFfLHjf4a678KtRtvEvh/UJ57GGXKXaoBJaMSQqyDkMpBC7sbWJKlRkBvd/hn4xfxx4Lg1a58hb5ZZIbqOBGVI3ByANxOfkKHqev4AA6TUtW03RrdbjVNQtLGBnCLJdTLEpbBOAWIGcAnHsaj0zXdH1vzf7J1Wxv/Jx5n2S4SXZnOM7ScZwevoa+WNbtbjx98cr7R9W1r7Mr6hcWUFxMoZYUjZxHGq5UckBQARlmzySc9v40/Z+sNN8L3V94Zn1W81K3xILWZo5POTPzBQqqdwHzDrnbgAkjAB7ppurabrNu1xpeoWl9ArlGktZllUNgHBKkjOCDj3FfNn7R3/JQ9P/AOwVH/6Nlro/gDpF9/wgvirUdLvI4dQvH+yWvmJ8sMscRZHJ5yN0w42n7vfOK8c8Y6L4l0TXPL8VrP8A2ncRLMXnuVnd05RSXDN/cI5PagD3/wDZx/5J5qH/AGFZP/RUVdn4E/5mX/sPXX/stfNvh7w18SU8L2GpeGptVTTdSu3jji0+8dPnBC+Y6qQFUlSu9umznA2k+/fCCG+tvCt9BqhkOoR6jIl0ZJN7GUJGHy2TuO7POTmsp/HH5ndh/wDda3/bv5noNFFFanCcl4f/AOSheMf+3L/0Ua+PdC0z+2/EOmaT53k/bruK283bu2b3C7sZGcZzjIr3P4j6R41v/GWv3ng+6vo/snkG7hsbpopJFMAKkKCN+NjDAy2WGAcnHhGlaVfa5qlvpmmW0lzeXD7Iok6sf5AAZJJ4ABJwBWVH4fm/zZ3Zj/GX+Cn/AOkRPvOvP/jb/wAkh13/ALd//SiOsPw34V+MGk3umxX3i/TbnS4bhGuY3YzSyRb8uu94dxJGQMtxwARitz42/wDJIdd/7d//AEojrU4TyD9nH/koeof9gqT/ANGxV9P18UeAPEXiDw14oS68NWn22/mieE2v2dpvOQjcRtX5uNobgj7vpkHuPE/w5+Kuo+HJtW16/nv8S+cdJW5kuJQ5fblIkBjGAxPynhSenSgD6forxP4R/Fa+1C/g8F+Jbe7fWEd4Ybp1wxEaMzLOGIIddhGcEnuAQSdj4xaT4/vLe3u/COp3Ys4UJubKyfyp9yhj5iuCGcEHb5YPUKQGJ4APVKK+IP8AhO/GH/Q165/4MZv/AIqvtPSY76HRrGLVJo59QS3jW6ljGFeUKN7DgcFsnoPoKALlcl4f/wCSheMf+3L/ANFGvknxhp2taT4sv7DxDcyXOqQuqzTvOZjINoKNvPJBXbjPIGAQMYr6C+A0GpWtlq1vq0V3DeRJbo0d2rLIijzAgIbkALtAHpjHFZVPij6/ozuwn8Gv/gX/AKXA9hooorU4Tkv+avf9wH/2vXW14n8YL3xTZ+LIv+ESS+a+m0wQzfYYDLIIS8m4jAJXkL8wwQcYIrxxPip45js7W1Hia+MdrKJoyzAuzAk4dyN0i8n5WJXoMYArKl9r1O7Hf8u/8CPs+ivmjWfi9498b74vCGkX1laxRGO6WwiN1IS+QCZAmY+Adu0A5ycnjHHzeKviZ4UvLa6v9T8R2UjbvJGpGUpJgYb5Jcq2Nw7HGQeuK1OE+x6K87+FPxMXx9pc8d+LS31q2cmS3gLAPFxtkUNk4ydpGTggE43AVqfEjx1b+A/C8t9mCTUpf3djaysR5r5GTgc7VB3Hp2GQWFAHYUV8yabqvxu8ceH2udOubuXT5nKCeL7NaMxUgnY/yPjIwSpxwR6ijwv8afFfhTW00zxmt3cWcSKk0U9qFu4gEOzGShJYlCS+SRyOTyAfTdeL/tI2NvJ4N0m/aPN1DqHkxvuPyo8blhjpyY0/L3NewWF9b6np1tf2cnmWt1Ek0L7SNyMAVODyMgjrXl/7Q32z/hWsf2bz/J/tCL7V5Wdvl7Xxvx/Dv2deN23vigDkP2aLG3k1HxFftHm6highjfcflRy5YY6cmNPy9zX0PXz/APsy/wDM0/8Abp/7Wr2jxP4n0vwjoc2r6vP5VvHwqry8rnoiDuxwfyJJABIANiivmzW/jV408V6z9n8Dafd20ECOxjhtVu55V3YDuNjBAAVGB0LHLNkY59/ir8UPDFxaWmqXl3CYkRlttSsFVpYwcfMWQOwO0gtnJ55zzQB9Df8ANXv+4D/7Xrra8i+Gvjf/AIT3xguqyWv2a6i0Y21yinKGRZlYsnOdpDA4PI5HOMn12sqX2vU7sd/y7/wIK5Lx3/zLX/Yetf8A2autrkvHf/Mtf9h61/8AZqK3wMMt/wB6j8/yZ1tFeH/Gb4t3mg3l14U0L9zeGJPtN+rkPBuBJRBjhipQ7wTjcQMMMjzDTvh/8QviDBY6yyz39rcfuo7++v1fYiuVOQzFwobccAHvgHNanCfX9FfHEN549+E2sW8covtOVZTKLWVy1pcnau7hTsk+UqCQcjjkEDH0foPxKs/E/ga81/RdPnvdQs4sz6RE485ZMcLk9VOCQwBJAOFLApQB3FFfMH/DR3jD/oG6H/34m/8AjtbGv/tH3FzocMeg6T9i1OTeLiS5YSpCOQpjxjc3IOWUAYxhs5AB9D18+ftMzwtceGrdZYzOiXLvGGG5VYxBSR1AJVgD32n0rb+C/wAUtd8Yatc6HriwXEkFo1yl4iCN2xIAVZR8p++MEBcbeck5GB+0vY28eo+Hb9Y8XU0U8Mj7j8yIUKjHTgyP+fsKAL/7NFjcR6d4iv2jxazSwQxvuHzOgcsMdeBIn5+xr3ivjzwV8QvF+g6NN4Z8LW0ck93cNcI8Vq09wG2ruCLypG2PnKnjJ+m5dfEz4v6D5F7q/wBugtRKoxfaSkUcp67C3lqeQD0IOM4IoA968d/8y1/2HrX/ANmrra8D8OfFL/hOLPw5pWprs1201m1kkdEwlzGCV8wY4VskAr05yvGQvvlZQ+OXyO6v/utH/t78wooorU4TyxbOzXwn8QNVaws5b60vNRkgnntklZCke5fvA8A84PFfNmhTavqHiHTLKzviLq4u4ooTcMXjDs4C71IIK5IyCDkdjXo3xH8cTWNr4m8GRWMZS/1aS6lumkOQol+4q467o1O4k8EjHceUWF9caZqNtf2cnl3VrKk0L7QdrqQVODwcEDrWFGEXBXR6mYYitHEyUZtLTq+yPuH/AIRbw9/0AdL/APAOP/Cj/hFvD3/QB0v/AMA4/wDCvFfBPx48Sa94t0/SL3Q7G4jvZVhH2JZEeLLDdIcl8qq7iRgdM5AFeqePPiBpHgHS1uNQ8yW7nRzZ2qKczsu3I3YIUDcpJPbOATxWvJHscX1qv/O/vZrf8It4e/6AOl/+Acf+FH/CLeHv+gDpf/gHH/hXy1eeOfiV461Se+0uXWQIkRGt9DE6xQjnGQhJySGOWJJ6dAAI9M+J3xA8F6xLFe319LMMedY6yJJP4TtyHIdOGDfKVzxnIo5I9g+tV/5397PofUtK07TPiF4S+wWFraeZ9s3+RCse7EQxnA5xk/nXcV5NofxA0jx94u8IXGn+ZFdwJdG8tXU5gZojgbsAMDtYgjtjIB4r1mopJKUku/6I6cbKU6VCUnd8j/8AS5hXJfEz/knuqf8AbL/0aldbXJfEz/knuqf9sv8A0alFf+FL0ZOV/wC/Uf8AHH80dbRXgfxP+ON9pus3Wg+FfLie1cw3N9LFuYSqw3LGrcYGCpLA5ycAYDHzz/hKfirp/wDxUsl54jjtW/fC4nikNoRJwCFYeVtO4beMDIx2rU4T6/r5/wD2mv8AmVv+3v8A9o1c+CfxI8UeLPFV/peu30d5AtkbiNjAkbIyui4GwAEEOc5B6DGOc3P2kbG3k8G6TftHm6h1DyY33H5UeNywx05Mafl7mgDlP2bNNml8VazqitH5FvZC3dSTuLSOGUjjGMRNnnuOvb6Tr44+HfxE/wCFe/2tPBo0F9fXkSxwTyybfIxuJ4CkspJUlQVzsHPQjUm+PPjyXVBdpfWkMAdW+xJaIYiBjK5bL4OOfmzycEcYAPrOivM/hT8Vl8fJPp+oW8dtrVuhmZYFbypYtwG5cklSCyggnuCCckLH8R/jB/wr/wAQ2+k/2F9v860W5837X5WMu67cbG/uZznvQB6hRXzpoH7SGpDVFHiPSrRtPKEE6dGyyq3Y4dyGHYjjrnPGDl+J/wBoPxLqF5NH4fWDS7ES5gkaFZJ2QDGH3bk5POAvHAycEkA+n6K4v4eePofGfgptdvIo7B7R3ivSzgRKyKrM6knhNrA89ORk4yfM/E/7R7xXk1t4Y0mCWGOXCXl6zETIByRENpXJ6Et0HIBOAAfQFFfJknxX+KsOlw6pLqN2mnzPsiu20yERO3PCv5eCflbgHsfSvX/hn8ZbHxeh0/XGtNO1oOBGobZFchmwoj3EnfkgbMknqM8hQD1SivH/ABp8eLfwp4outEt/D8961rhZZZpzb/ORnCqUJK4Iw3Gc5GRgnnPDfx/17WfGum6bPpGmpp99epbhU3+bGsjbVO8tglcgn5RnB+7ngA+g6K4v4k+Pm+HujWmojR5NQS4uPIOLhYljO0sMnBJJ2nGFxwckcA+Kf8NHeMP+gbof/fib/wCO0AfT9FYfg7XJvEvg7StZuLWS2nu7dXkjeMp83QlQSTsJG5TnlSp71j/ED4k6R4BsMztHc6pIge308OVaVd4UksFYIACxBbrtIHQ4AO0ry/4/an9g+F89t5Pmf2hdw227djy8Ey7sY5/1WMcfez2wfHI/i/8AES/eZdFkjtbS1t9/2PTtNjaK1hRQCcMrFUHGSTgZ7DAo8Q/GrWvFHgObw3qdjaNPO8YlvkypdEKt9zoHLKCWBAwSAo60Acn4E/5KH4a/7Ctr/wCjVr6y8P8A/JQvGP8A25f+ijXyb4E/5KH4a/7Ctr/6NWvrLw//AMlC8Y/9uX/oo1lU+KPr+jO7Cfwa/wDgX/pcDraKKK1OEKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigArJ8U/wDIoa1/14T/APotq1qyfFP/ACKGtf8AXhP/AOi2qZ/CzfC/x4eq/Mz7LWbPw98NLLV799lraaXFK+CAWxGuFXJALE4AGeSQK+VvC1i/xA+KtpHexwf8TPUHuruMMyIUy0sqqRlhlQwHPccjrWJruralqF4be91C7uYLSR0to5pmdYVzjCAnCjCgYHoPSn+FvEl54R8R2muWEcEl1a79iTqSh3IyHIBB6Me9EPhQYr+PP1f5n3PXi/7R+jfa/CWmauiTvJYXZibYMokcq8s3HHzJGAcgfNjkkV65pN5NqOjWN7cWklnPcW8csltJndCzKCUOQDkE46Dp0FfPHxl+KepXt7rPguLS47bT0dIpZLlGE7sjh9684CNhcZByvORuwKMDj/g5r7eH/ibpTbpPIvn+wzKiqxYSEBBz0Ak8skjnAPXofsOvgCvfLP8AaTm/4R+6W90OP+2lTFs8LH7O7Enl1J3KFGOAW3HPKdgDg/jb/wAle13/ALd//SeOvp/wJ/yTzw1/2CrX/wBFLXxp4j1++8U+ILzWtSaM3d04Z/LXaqgAKqgegUAc5PHJJ5r3v4OfEDxBc6dd6Jq2jz3FroulLPBLbW7CZkABji29GZ0I2Y25CZ+bJIAM/wDaa/5lb/t7/wDaNbH7N1jbx+DdWv1jxdTah5Mj7j8yJGhUY6cGR/z9hXjnxB+JOpfEO4tGvbO0tYLJ5TbJDuLBXK8OxOGICDkBe/Hp2HwM+Ic2h38PhFtKku4NSvd6TW5JkhZkCsWXoyAIpJ42gMfm6UAZ/wC0FZzW3xNM0t3JOl1ZRSxRtnECgsmxeTxuRn4xy547n2P4FX1vd/CjTYYJN8lpLPDONpGxzI0gHPX5XU8evrmsv43/AA5m8U6XFruj20k+sWKbHiRzma3G5iFXHLqxyAMEgsPmO0V4x4C+I2tfDTVLq1e1kns2dlutNnYxFZRxuBIJRwRg8cgYIyAQAfYdfGGu/wBnf8Lk1P8Atf8A5Bn/AAkEv2z73+p+0Hf935vu56c+lep61+0nD/ZcI0LQ5BqDoDKb5gYoW+XIUIcyD7wydnY4PIr58oA+0v8Amr3/AHAf/a9dbXi3wj8Z33jfxQL/AFOOMXltpLWksqcCYrKrb9uMKSHGQOMgkYBwPaaypfa9Tux3/Lv/AAIK5Lw//wAlC8Y/9uX/AKKNdbXJeH/+SheMf+3L/wBFGip8UfX9GGE/g1/8C/8AS4HAftB+CZtV0u18UafbyS3FghivFQFj9n5YP14CMWzgE4cknC1gfBv4qWPhzwnqmmeIb2NLbTkE9hEkf72UMzb417MdzKQDz87Enap2+/6tpsOs6NfaXcNIsF7byW8jRkBgrqVJGQRnB9DXxh4c8LtqHxBs/DGqR3cTm9NrdLaIsskZUkPjnGBg5bkKAWw2MHU4T0jTvg9Yy/BG48QX7SQa19nk1SGVTkCBU3LEy7ipDKN27hgXAP3SDz/wb+Idj4D1TVF1VJPsF7bhi0Me9xLHuKKOQMMGcfUrkgZNfV88EN1by29xFHNBKhSSORQyupGCCDwQRxivijxD4MvtC8eTeEopI728FxHBAyfIJTIFMf3jhSQ65ycA55I5oA9r+CfhW41TVr34k6p5CzalLc/Z7dYQQpeQFpUYsSvIlTBGcZ5wefcKz9D0az8PaHZaRYJstbSJYkyAC2OrNgAFicknHJJNR6/4j0jwtpbalrV9HaWgcJvYFizHoFVQSx6nAB4BPQGgD5c+Ot9cXfxX1KGeTfHaRQQwDaBsQxrIRx1+Z2PPr6Yr6X8Cf8k88Nf9gq1/9FLXyJ4+8Sw+L/G+p67b28lvBcugjjkILbURUBOOASFzjnGcZOM19L/Cnx3oHiDw1pGhWN1J/aljpka3Fs8LKVEYWMtuxtIJwRg5wRkDkAAw/wBo7/knmn/9hWP/ANFS1l/s0zwto2v26y3ZnS4id42YeQqsrBSg6hyVYMe4VPSsT41/Ezw14u8PWejaHcT3ckd2l00/kNHGAEkUr8+G3fMD93GD1zxR+zn4nt7HWNR8OXU84k1DbLZx8mPeiuZP91iu05xyEwTkKCAef/FT+0f+FoeIf7U/4+PtZ2fd/wBTgeT93j/V7Pf15zXt/wDwzj4P/wCglrn/AH/h/wDjVcx+0j4euF1HSfEqtutXi+wSDAHluC8i98ncGftxs68ir/ws+N1vJZx6J4xvfKuI9qW2pS5IlBIAWU9mGfvnggEsQRlgDY/4Zx8H/wDQS1z/AL/w/wDxqu08D/DzRfAFveRaS93K946tNLdSBmIUHao2gAAbmPTPzHJPGMfxL8aPCGg6NBe2l9Hq89yhaC2s5FLD5cjzc8xDJAORu5+6cHGP8Fm1O50bXfHXiS/jL6u6FppFjjURW6snmErgKPvLggY8vPOc0AeeftHf8lD0/wD7BUf/AKNlr1/4Jf8AJIdC/wC3j/0okrwj41+KdG8XeMrO/wBDvPtdrHp6Qs/lPHhxJISMOAejD869n+BWuaXd/DjTdIgv4H1K088z2u/EiAzMwbaeSuHX5hxk4znIoAz/ANo7/knmn/8AYVj/APRUteZ/DH/knvj3/uH/APo167T9ozxPpcuk2XhiKfzdTju0u5kTkQoI3ADHsx3ggegycZXPmfw91n7JpXijSHeBI7+0hlXecO8kU6YVeefleQkYJ+XPABrKv/Cl6M7sr/36j/jj+aPsSiiitThCvE/EH/Jr9z/wH/0tFe2V4trUE1z+zFdJBFJK4TeVRSxCrd7mPHYKCSewBNZP+KvR/od1P/can+OH5VDif2cf+Sh6h/2CpP8A0bFX0/XyJ8FfFFj4W+IMc2pSRw2l7bvZvcSPtWEsVZWPB43IF5wBuyTgV9T6b4l0HWbhrfS9b02+nVC7R2t0krBcgZIUk4yQM+4rU4Txf9pmCZrfw1cLFIYEe5R5Ap2qzCIqCegJCsQO+0+lbn7OP/JPNQ/7Csn/AKKirjP2gPGGi+IH0fTtG1K0vjaPK9y8CBwpZY9m2XGCCN2QrYyBu5UY6f8AZ11bTYfCF1pcuoWiahNqcrxWjTKJXXyY+VTOSPlbkDsfSgD0zx3/AMk88S/9gq6/9FNXzB8Ev+SvaF/28f8ApPJX0X8TPEmi6T4K16xv9UtILy40yZIbZpR5shdWRdqfeILcZxgYJOADXzh8GJ4bb4t6C88scSF5UDOwUFmhdVHPcsQAO5IFAHof7TX/ADK3/b3/AO0a7z4HQQw/CTSHiijR5nneVlUAu3nOuW9TtVRk9gB2rL+PXg648R+EodWsvmuNG82Z4y4UNAVBkIyOWGxSORwG6nArn/2dvGFu2nXfhO8udt0kpuLJZZSfMQj50QEYG0qXwDzvY44JoA94r5A+Nv8AyV7Xf+3f/wBJ46+q/Eev2Phbw/ea1qTSC0tUDP5a7mYkhVUD1LEDnA55IHNfJng/w9cfFP4lTJdN9nju5Zb+/e3AHloWy2wMe7Mqj72N2SCAaAPqfx3/AMk88S/9gq6/9FNXzB8Ev+SvaF/28f8ApPJX1H40gmuvAviG3t4pJp5dMuUjjjUszsYmAAA5JJ4xXzB8DjCPi3pAljkZyk4iKuFCt5L8sMHcNu4YGOSDnjBAPruvjD4qan/a/wAUPENz5PlbLs223duz5IEW7OB12Zx2zjnrX2XPPDa28txcSxwwRIXkkkYKqKBkkk8AAc5r4o8f61Y+IfHms6rpkEcNncXBMWwYEgAC+ZjAILkFyCM5Y5yeaAPo7wJ/yG/DX/YmWv8A6Etem15l4Fms5/EHh9rC5+02q+EYI0lIAJ2yBTuALBWBBBXJwQRnivTaypfa9Tux3/Lv/Agrkv8Amr3/AHAf/a9dbXJf81e/7gP/ALXoq9PUMD/y8/wP9Dw39o7/AJKHp/8A2Co//Rstev8AwS/5JDoX/bx/6USV4R8db64u/ivqUM8m+O0ighgG0DYhjWQjjr8zsefX0xXu/wAEv+SQ6F/28f8ApRJWpwnzhYJcah8ZLaPXLOAXVx4gQX1rgPGHa4HmJjJBXJI6nI7mvs+vij4j2Nxp/wASvEcN1H5cjahNMBuBykjGRDx6qyn2zzzX2nBPDdW8VxbyxzQSoHjkjYMrqRkEEcEEc5oA83+POpQ2PwrvLeVZC9/cQ28RUDAYOJctz02xsOM8kfUeWfs4/wDJQ9Q/7BUn/o2KvQ/j/wCKLHTvBD+HzJHJqGpvGRCHw0cSOHMhGDxuQKAcZySM7SK5T9m7w9cNqOreJWbbapF9gjGAfMclJG75G0Knbnf14NAHb/HTwj/wkfgZ9St1zfaNuuU5+9CQPNXlgBwA+cE/u8D71eUfAXxjb+HPFs2k3vy2+s+VCkgQsVnDERg4PCnewPB5K9Bk19T18QeKtHuPBnjnUdNgeeCTT7sm1l80eYEzuifcvRipVuMYJ6A8UAez/HfxPe3+rab4D0SfzJr3aLy3HlkSO8iGBCx5RgV3H7vDqckHj2Tw5oFj4W8P2ei6asgtLVCqeY25mJJZmJ9SxJ4wOeABxXi/wR8Nal4h8QXXxG8QXElxO7yRWrShg0khAVpB0XYFLRgDI6jC7BXvlAHwhrup/wBt+IdT1byfJ+3Xctz5W7ds3uW25wM4zjOBXu9rcftBW/n+bZQXXmRNGvnGzHlMejrsZfmHbdleeQa8k8YWF94J+Jt/FFfSTXljerdQXUrea7EkSxu5YYZ8MpbIwTnqK+t/B/iWHxh4TsNegt5LdLtGJhcglGVijDI6jcpweMjHA6UAeKX3hv44+LrOTQ9ckgj026x5zztahBtIdcmIF/vKOg+vGa7P4L/DzWvAlvqsusvaB9RS3ZIoZC7RFQ+5X4Az84Hykjg8+vpGralDo2jX2qXCyNBZW8lxIsYBYqiliBkgZwPUVw/gT4kzeJfAuueKdUs44INOuLgrBa5ZvJjiWTBLHDPhiM/KDxwKAPBPin8N7zwNrklxDDv0K7lY2cyZIizkiFskkMB0JPzAZ67gOs8E/tB32lW8Gn+KLSTUbeJAi3sLf6RgBvvhjiQ/dGcqcAk7ia9H+FnxDX4kaNqVnriab/aCOyyWMUbBZLZlUbirltwLFlPOBxkDIzx/xh+E3hzSfDF/4p0ZJNPnt3jaS1jOYJN8m04U8ocuOh2gLgKM5oA9w0rVbHXNLt9T0y5jubO4TfFKnRh/MEHIIPIIIOCK+VPjrfXF38V9Shnk3x2kUEMA2gbEMayEcdfmdjz6+mK7/wDZovriTTvEVg0mbWGWCaNNo+V3DhjnryI0/L3NecfG3/kr2u/9u/8A6Tx0AfR/wrt/s3wv8PR/YfsWbQSeV5vmbtxLeZntvzv2/wAO7b2p/gT/AJmX/sPXX/stXfAn/JPPDX/YKtf/AEUtUvAn/My/9h66/wDZayn8cfmd2H/3Wt/27+Z1tFFFanCebapIsNx8UZXhjnRNMjZopCwVwLV/lO0g4PTgg+hFeA/B+xt9Q+K+gQ3UfmRrK8wG4jDxxvIh49GVT7454r37VJ5rW4+KNxbyyQzxaZG8ckbFWRhauQQRyCDzmvnn4V3VnZ/FDw9LfW/nwtdiJU2BsSOCkbYP912Vs9RjI5ArKj8Pzf5s7sx/jL/BT/8ASIn2fXn/AMbf+SQ67/27/wDpRHXoFeZ/Hl4V+Fd4JbKS4d7iERSLEHFu28Hex/gBUMmR3cD+KtThPPP2a9M83xDrmredj7NaJbeVt+95r7t2c8Y8nGMc7u2Ofo+vnz9maeFbjxLbtLGJ3S2dIyw3MqmUMQOpALKCe24etfQdAHxxYTXFx8dbaa8tfsl1J4lR5rfzBJ5Tm5BZNw4bByMjrivseviDwJ/yUPw1/wBhW1/9GrX2/QB8UfEPww/hHxzqel+R5NqJTLZgbipgY5TDNy2B8pPPzKwycV9d+HPFFj4h8HWfiRZI7e0mtzNKZHwsBXIkBZgOFZWG7ABxnpXnf7ROifbvA1pq0dvvm027G+Xfjy4ZBtbjPOXEQ6Ej6ZrwTS/HOtaP4O1XwvaSxrp+pOHlOCJEPAbawI4ZVCsDkEccZOQDrPhvY6p8Q/jDFr19HPJHDd/2heTxN8kJXLRJls/LuVVC8naDjhSR9BeH/wDkoXjH/ty/9FGsz4PeDf8AhEPA1v8AaYfL1PUMXV3uXDpkfJGcqGG1eqnOGZ8da0/D/wDyULxj/wBuX/oo1lU+KPr+jO7Cfwa/+Bf+lwOtooorU4Tkv+avf9wH/wBr18q/EexuNP8AiV4jhuo/LkbUJpgNwOUkYyIePVWU+2eea+qv+avf9wH/ANr182/G3/kr2u/9u/8A6Tx1lS+16ndjv+Xf+BH0/wCBtE/4RzwNouktb/Z5oLRPPi379szDdJzk5+cseDj04xR428MW/i7wlqGkTQQSzSRMbVpsgRThT5b5HIwTzjsSMEEg6GhWdvp/h7TLKzM5tbe0iihNwhSQoqALvUgENgDIIGD2FSatqUOjaNfapcLI0FlbyXEixgFiqKWIGSBnA9RWpwnyR8HNfbw/8TdKbdJ5F8/2GZUVWLCQgIOegEnlkkc4B69D7n8YPhxrHxA/sb+ybmxh+w+f5n2t3XO/y8Y2q39w9cdq8I+Efh648Q/ErSUgbZHYSrfzvgHakTAjgkZy21eM43ZwQDXoHx5+IWqW+uS+E9Kv/JsRaKNQSOPa7u/zbC5H3dmz7uM72BJ6AA9H0G80L4T+BNO0TxH4gsY7q0iMkiqxLt5krH5Ixl2UMxG4LztJIHIHgnxn8W6R4x8aw3uizST2kFlHb+c0ZQOwZ3JUNg4+cDkDkHtgn0fwH8AtIfRrDU/FJu57yZPNewVjDHGrL8qPwH3jOTgrg8YIGT558cLDQtM+Ir2eh2kFosdpF9qhgjMaLMcnhegyhjPy8c+uaAPof4V6Z/ZHwv8AD1t53m77QXO7btx5xMu3GT034z3xnjpWX8cZ4YfhJq6Syxo8zwJErMAXbzkbC+p2qxwOwJ7V0ngT/knnhr/sFWv/AKKWvP8A9o7/AJJ5p/8A2FY//RUtAHP/ALMv/M0/9un/ALWrnP2hdfbUvHkWjq0nkaVbqpRlUDzZAHZlI5IK+UOe6nA7noP2ZoIWuPEtw0UZnRLZEkKjcqsZSwB6gEqpI77R6Vwfxt/5K9rv/bv/AOk8dAHWfCv4p+EPA/g6W1vNLu01Rrj9/JaosjXSncVYliAoQfLtz3BGdzYr/GL4raL410a30XRre7ZIL0zvdTKEVwqsi7FySQ28n5tpGBxzx2/w++HPgfxd8PNC1W+8NxrcG3aKRlupVMjJI6s7bWXJZgTz0BCg4UV0n/Ckvh5/0L3/AJO3H/xygDyP4A6n9g8UwW3k+Z/aDTW27djy8RiXdjHP+qxjj72e2D9OV4t4JvfCmjfFefR/DmmXdlARcaa6OxcG5jJd3yzsdhWLA75xwOTXtNZUvtep3Y7/AJd/4EFcN8UdSh0bRtI1S4WRoLLVobiRYwCxVFdiBkgZwPUV3NcF8WNM/tvw9puk+d5P27U4rbzdu7ZvR13YyM4znGRRW+Bhlv8AvUfn+TPj2vv+vgzSdSm0bWbHVLdY2nsriO4jWQEqWRgwBwQcZHqK+74J4bq3iuLeWOaCVA8ckbBldSMggjggjnNanCeX/tBabNffDI3ETRhLC9iuJQxOSpDRYXjrukU844B+h88/ZuvriPxlq1gsmLWbT/OkTaPmdJECnPXgSP8An7CvS/jzqUNj8K7y3lWQvf3ENvEVAwGDiXLc9NsbDjPJH1Hmn7N1jcSeMtWv1jzaw6f5Mj7h8rvIhUY68iN/y9xQBX/aO/5KHp//AGCo/wD0bLXX/BX4eaFqXw6l1HW9Jsb6TVJZBHJIpZ44VzHgE/6ttwkOU55XnIGOQ/aO/wCSh6f/ANgqP/0bLXtfwl1KbVfhX4fuJ1jV0tzbgICBtidolPJPO1AT756dKANDw14B8L+ELie40LSY7WedAkkhkeRtoOcAuxIGcEgYzgZ6CvIP2mZ4WuPDVussZnRLl3jDDcqsYgpI6gEqwB77T6V9B18wftHf8lD0/wD7BUf/AKNloA6/9m7RrNPD2ra5s3X0t39j3sAdkaIj4U4yMl+ecHavHFeyarpVjrml3GmanbR3NncJslifow/mCDggjkEAjBFeX/s6iEfDm6MUkjOdTlMoZAoVvLj4U5O4bdpycckjHGT65QB8P+ENSm0bWW1S3WNp7KL7RGsgJUsjKwBwQcZHqK+4K+I7WX+2/EOsy6dp3k/blna3sbdd2ze/yxoABnGQoAA+lfblZQ+OXyO6v/utH/t78wooorU4Tw/xvBDN8JPHjyxRu8PiB3iZlBKN50S5X0O1mGR2JHevLPgxBDc/FvQUnijlQPK4V1DAMsLsp57hgCD2IBr0P4jyWKfDLxMt3DJJO/iuRbNlPEcuMlm5HHliQd+WHHccB8Ev+SvaF/28f+k8lZUfgR3Zl/vUvl+SPr+vjD4pazea38StdlvHz9mu5LOFATtSOJiigAk4zgsccbmY4Ga+z6+DNW02bRtZvtLuGjaeyuJLeRoySpZGKkjIBxkegrU4T7P8AaA3hfwHo2jyrIs8FuGnR2VikrkvIuV4IDMwGM8AcnrXH/HzQLHUvh5NrE6yfbNKdGtnVsDEkiI6sOhBGD65Uc4yD6RpMl9No1jLqkMcGoPbxtdRRnKpKVG9RyeA2R1P1NcP8cZ4YfhJq6Syxo8zwJErMAXbzkbC+p2qxwOwJ7UAeH/BG+t9P8d2s11J5cbSrCDtJy8ivGg49WZR7Z54r60r5D+D2mf2v40s7bzvK2XcFzu27s+STLtxkddmM9s556V9eVlT+KXr+iO7F/waH+B/+lzCuS+Jn/JPdU/7Zf8Ao1K62uS+Jn/JPdU/7Zf+jUor/wAKXowyv/fqP+OP5o+Nb+xuNM1G5sLyPy7q1leGZNwO11JDDI4OCD0r3/4b/GpNe8rwz4zh+03V9L9miuhApjnEmR5cqDgZJCggEEMNwGCx6DXvBXw6+KOp6jJp2rwHX+GlubK780nbGqqTGSVaMbo8lMcjG4HNeEeLvhj4p8F7pdSsfNsRj/TrQmSH+HqcApywX5guTnGa1OE93+GXwivPAHi3UNTm1aC8tZLT7NbhIijtuZGZmBJC4KYABbOc8YxVf9o7/knmn/8AYVj/APRUtch+z540vIdcbwleXe6wniklsonUsUmHzMqkfdUrvYg8ZXjBJ3dP+0jfW8fg3SbBpMXU2oedGm0/MiRuGOenBkT8/Y0AcJ+zxpS3vxBuL+W2kkSwsneOYbtsUrlUGSOMlDLgH0J7ZH0P4yg0258Fa2msRSS6eLKV51iVWcKqlspu43jGVJ6EA14x+zNPCtx4lt2ljE7pbOkZYbmVTKGIHUgFlBPbcPWvY/Hf/JPPEv8A2Crr/wBFNQB8qfCXUodK+Knh+4nWRke4NuAgBO6VGiU8kcbnBPtnr0r6z8Q+FdC8V2Ytdc0yC9jX7hcEPHkgna4wy52jOCM4weK+TPhLpsOq/FTw/bztIqJcG4BQgHdEjSqOQeNyAH2z0619V+O/+SeeJf8AsFXX/opqAPizSdNm1nWbHS7do1nvbiO3jaQkKGdgoJwCcZPoa+o7r4A+BrjToLaK3vrWaPbuu4bomSXAwdwcMnJ5O1RyOMDivmjwnfW+meMtDv7yTy7W11C3mmfaTtRZFLHA5OAD0r7noA+dPjZPaeD/AAnoHw+0SWRLRUa5uUdnLuu47CzcKwZzIxXHBRSAoAq/+zl4XsZLLUPFE8cct4lwbO23JzAAgZ2U5xlhIB0yAp5wxFc5+0d/yUPT/wDsFR/+jZa7v9m+eFvAup26yxmdNTZ3jDDcqtFGFJHUAlWAPfafSgD2SvhzxlpS6H411vTIraS2gt72VIIn3ZWLcfL+9yQV2kE9QQec19x18UfEe6+2fErxHL9ngg26hNFsgTap2MU3Ef3m27mPdiT3oA9/8c2//Ca/A2LxBeaHBLq0WnrfRLI2zydygyyIVf7uzc4Usc4TcpI2188eBP8Akofhr/sK2v8A6NWvqPVYW0P4D3FnqZjtp7fw59llV5Fwsv2fZsznBJbCjB5JGM5r5c8Cf8lD8Nf9hW1/9GrQB9v18WfE6Oxi+JviJdOmkmgN67MzjBEpOZV6DgSFwPYDk9T9p18UfEe+uNQ+JXiOa6k8yRdQmhB2gYSNjGg49FVR7455oA+r/Cd9b6Z8LNDv7yTy7W10S3mmfaTtRYFLHA5OAD0r401bUptZ1m+1S4WNZ724kuJFjBChnYsQMknGT6mvruwsbjU/gVbWFnH5l1deGkhhTcBudrYBRk8DJI618seBP+Sh+Gv+wra/+jVoA+y9A8OaR4W0tdN0WxjtLQOX2KSxZj1LMxJY9Bkk8ADoBXj/AO0V4c0iHQbXxFFYxpq017FbS3KkgvH5chwwzgn5V+YjOABnAxXuleP/ALR3/JPNP/7Csf8A6KloA8A8Cf8AJQ/DX/YVtf8A0atfWXh//koXjH/ty/8ARRr5l+D99b6f8V9AmupPLjaV4QdpOXkjeNBx6syj2zzxX014f/5KF4x/7cv/AEUayqfFH1/RndhP4Nf/AAL/ANLgdbRRRWpwhRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFQ3VrDe2c9pcJvgnjaORckblYYIyOehqaigabTujkv+FZ+EP+gR/5My//ABdH/Cs/CH/QI/8AJmX/AOLrraKy9hS/lX3Hb/amO/5/T/8AAn/mcl/wrPwh/wBAj/yZl/8Ai6P+FZ+EP+gR/wCTMv8A8XXW0Uewpfyr7g/tTHf8/p/+BP8AzOS/4Vn4Q/6BH/kzL/8AF0f8Kz8If9Aj/wAmZf8A4uutoo9hS/lX3B/amO/5/T/8Cf8Amcl/wrPwh/0CP/JmX/4uj/hWfhD/AKBH/kzL/wDF11tFHsKX8q+4P7Ux3/P6f/gT/wAzkv8AhWfhD/oEf+TMv/xdH/Cs/CH/AECP/JmX/wCLrraKPYUv5V9wf2pjv+f0/wDwJ/5nJf8ACs/CH/QI/wDJmX/4uj/hWfhD/oEf+TMv/wAXXW0Uewpfyr7g/tTHf8/p/wDgT/zOS/4Vn4Q/6BH/AJMy/wDxdH/Cs/CH/QI/8mZf/i662ij2FL+VfcH9qY7/AJ/T/wDAn/mYejeD9B8P3j3el2P2ed4zGzec75UkHGGYjqBW5RRVxioq0VY5atapWlz1ZOT7t3f4hXJXfhTWP+Eh1HVdK8Sf2f8Ab/K8yL7Ckv3E2jlj9T0HWutoolBS3LoYmpQbcLaqzuk1a6ezTW6RyX/CP+L/APoeP/KTF/jVOPwTrsOqTapF4otE1CZNkt2uh24ldeOGfqR8q8E9h6V3NFR7GPn97/zN/wC0a3aH/guH/wAicl/wj/i//oeP/KTF/jVOTwTrs2qQ6pL4otH1CFNkV22h25lReeFfqB8zcA9z613NFHsY+f3v/MP7Rrdof+C4f/InJf8ACP8Ai/8A6Hj/AMpMX+NU9V8FeIdc0u40zU/F0dzZ3CbJYn0mLDD88gg4II5BAIwRXc0Uexj5/e/8w/tGt2h/4Lh/8ieJ/wDDOWl/9Bf/AMlm/wDjtb+g/Ci68MajeX+jeIYLO6u+JXTSozxnO1QWIRc/wrgcDjgY9Noo9jHz+9/5h/aNbtD/AMFw/wDkTx7UvgLZarcLPcanaI6oEAtdLS3XGSeVjdQTz1xnp6CpNE+ByeHNYg1bSdf+z30G7y5fse/buUqeGkIPBI5Feu0Uexj5/e/8w/tGt2h/4Lh/8icNqvgrxDrml3Gman4ujubO4TZLE+kxYYfnkEHBBHIIBGCK4z/hnLS/+gv/AOSzf/Ha9soo9jHz+9/5h/aNbtD/AMFw/wDkTxaD9nnTba4inTVoy8bh1ElkXUkHPKtIQw9iCD3rtL/wl4n1PTrmwvPGfmWt1E8Myf2XENyMCGGQ2RkE9K7Wij2MfP73/mH9o1u0P/BcP/kTxP8A4Zy0v/oL/wDks3/x2r+ifA5PDmsQatpOv/Z76Dd5cv2Pft3KVPDSEHgkcivXaKPYx8/vf+Yf2jW7Q/8ABcP/AJE8i1v4HJ4j1ifVtW1/7RfT7fMl+x7N21Qo4WQAcADgVUt/2e7C1nWaHWdsi5wfsrHGRjvJXtFFDowas7/e/wDMqOZ14SUoqKa/uQ/+RCiiitTzwrzzwzL4v8OeHrXSv+EQ+0eRv/e/2lEm7c5bpzjrjrXodFZyhzNNOx10MV7KEqcoKSbT1vur22a7s+Y9Z+But3eovNpGiz6favk/Z5r6G42EknCtlTtAwADk8ck5rpPAnw11XwVqltrDeFru/wBUgR1WRtVgjiUtkblQAnO0leWI5Jx0x7xRS9nL+Z/h/kX9bo/8+IffP/5M+a4fgbeLeXLTaBqr2rbfs8aataq8fHzbmKEPk9MKuOnPWtTSfhFNo2s2OqW/hfWWnsriO4jWTXLUqWRgwBxEDjI9RX0BRR7OX8z/AA/yD63R/wCfEPvn/wDJngfxE+HHiPx3rkWrw+G/7MuvKEVxi7glE2PuscFTuA4JJOQFHGOeP/4UL4v/AOfb/wAiRf8Axyvqyij2cv5n+H+QfW6P/PiH3z/+TOOg1zxlDbxRP4OkndECtLJqsAZyB947VAyevAA9AK8k8Q/CDW77XBq3h7w/PoM3m+cY4tQhdI34KmLBUx4IJ6nGRjaBivoyij2cv5n+H+QfW6P/AD4h98//AJM+Y9T+EXxH1vyv7WvL6/8AJz5f2u8SXZnGcbpTjOB09BXpfgzRtV8C6W9jo/gKQmV981xPq8DSzHnG4hQMAHAAAA5PUkn1Gij2cv5n+H+QfW6P/PiH3z/+TOS/4SDxf/0I/wD5Vov8K8c8UfBnVdZ1R77R/DUmjiZ2eW3F9BLECcf6sZUoM7jjJHOBtAxX0hRR7OX8z/D/ACD63R/58Q++f/yZ8z6l8KfibrNutvqmpalfQK4dY7q+WVQ2CMgNKRnBIz7mqd18CPEb+R9j0+eLESibzrqCTfJ/Ey4ZdqnjCncR/eNfUlFHs5fzP8P8g+t0f+fEPvn/APJnjvwf+H2veDNZu5NUttkElu6rJ5iH5i0fGFYnop5r2KiiqhDlVr3McTiHXkpcqjZJJK9rL1bf4hXk/wAUdN1S+1LUYbC11Q/bdDFolxZW7SAP5+8oxBGFZVKt14foeh9YoonDmW48LiFRk243TTVttz4z/wCFV+MP+gNd/wDgLN/8RXoHw1h+IHgvWLaK9tdal8PDd51jHaySfwvt2B1Gz52DHaVz3zX0ZRU8k/5vwNfb4X/nz/5Mzxr4iaZpfxBs4jL4Z8UWWpwYEN8mlbyEzkoy7xuXkkcjB5B5YHgPDen/ABa8I6dJYaHHd2lrJKZmT+zTJlyACcvET0UflX1JRRyT/m/APb4X/nz/AOTM+V7P4e61rviC61Xxvb+JJnmffIbCwLSSkgj7zgBAMLgBWGBgbcCvc7Dxbb6Zp1tYWfhDxRHa2sSQwp9gJ2ooAUZLZOAB1rtaKOSf834B7fC/8+f/ACZnJf8ACd/9Sp4o/wDBd/8AZV5J8WfDtz431Sx1PQvCmrWt4EdL2W4sXQzAbfL+7uBIAYZODjaOQBj6Ioo5J/zfgHt8L/z5/wDJmebeD9Vh8H+E7DQYPDniu4S0RgZn00AuzMXY4DcDcxwOcDHJ61uf8J3/ANSp4o/8F3/2VdbRRyT/AJvwD2+F/wCfP/kzPC/ihoVv8QPJv7bQPFFlq9tEYkd9NLRzINxVGG75fmP3hnAJyG4x5vpngn4j6J5v9kprVh52PM+yC4i34zjO1RnGT19TX15RRyT/AJvwD2+F/wCfP/kzPkubwN48168tl8SyeIbi1i3Yke3nuniyP4VcqOSFz8w9ecYr2ufVYW8Cy+Frfw54rjgOmHTo55NNDsq+V5YYgMATjntn2r0mijkn/N+Ae3wv/Pn/AMmZ8h6J4G+IHhzWINW0nTru3voN3ly/YpH27lKnhoyDwSORW54zsfil46dF1fT7hbSJ98Vnb2UyRI20DONpLHryxONzYwDivqCijkn/ADfgHt8L/wA+f/JmeRfDuKz+H2hy2MPh3xReXVxKZbi7OkiMv2VQAxIUDoCTyzHjOK5j4teH7zx5qNlq2k6H4hgvoYhbSRXdgRG0YLMGBXJDAsRgjBBHTHzfQlFHJP8Am/APb4X/AJ8/+TM+R9N8IfE3RrdrfS212xgZy7R2puYlLYAyQqgZwAM+wr3v4SWWqWPhKdNZiukvnvGeRrpWDyExx5c7uTkg89+a72ihU5cybd7BPFUvZSp06fLzW6t7BRRRWpwnJeH/APkoXjH/ALcv/RRr518dfB3xB4Z1wxaRYX2r6ZNl7ea2gaV0H9yQKOGGRzgBuoxyq/RXh/8A5KF4x/7cv/RRrrayo/D83+bO7Mf4y/wU/wD0iJ86fDDSfilqPjG1udU1PxBY6bZOJbn+1HmKzryPLWOQ4YsMjP8AD1znaCfGy18f6z4nudJg0zUr7w6rxXNmtrY+aoby9rEuils7jJ8rHuDjGK+i6K1OE+QPDEfxR8Hfav7A0XXLP7Vs87/iUNJu252/fjOMbm6ete3/ABSuviLbeEtNXw7b77p4iNXl0xN7qxVVxCG+faWZzlRuG0HK859QooA+KLDwl440zUba/s/DGuR3VrKk0L/2bKdrqQVOCuDggda+v/CuoapqvhfTr7WtO/s7Upog09rn7hzwcHlcjDbTyucHkGtiigCvf2NvqenXNheR+Za3UTwzJuI3IwIYZHIyCelfHHwx8I/8Jp45stNlXdYxf6Te84/coRlfvA/MSqZByN2e1fR/xk8T2/h34dalCZ4FvtSia0toJMkyB8LIQB/dRmOTwDtz1ANP4I+DF8L+Co9QlkjlvNZSO7dk3YSIrmNOTgkBmJIA5YjkAGgD0yuS8P8A/JQvGP8A25f+ijXW1yXh/wD5KF4x/wC3L/0UayqfFH1/RndhP4Nf/Av/AEuB1tFFFanCcl/zV7/uA/8AtevmH4tR30XxU8QLqM0c05uAysgwBEUUxL0HIjKA+4PJ6n6e/wCavf8AcB/9r18m+O/+Sh+Jf+wrdf8Ao1qypfa9Tux3/Lv/AAI9j8JfFrWvCfhrTF8X+GLuPQzZQxabe2VuQXCgqofe+0llQsOVOBkKVYEcx8SfibN8UBaeHfD+hXbQJcfaI8qZLiZljIwI0yFADSZ5bIAPy4Ir6fnghureW3uIo5oJUKSRyKGV1IwQQeCCOMVHY2FnplnHZ2FpBaWsedkMEYjRckk4UcDJJP41qcJ538IPhm3gXS5r7UzG+tXyKJVUKwtkHPlhupJOCxBwSFAzt3HzT9oHwbeWnij/AISi2hnmsb6JftUiqWW3kQLGNxC4VWGzGSSW3e1fS9FAHyx4V+NXji007TvDulaXY6nJbxCG3X7JLJM6IOBiNxnaoxwOi5OTk1zfjbwf48sbifXvFem3bPcuXmvN6SqDlVG4xkhByqqDgdh0wPsuigD5/wD2eL/xTfajf/aru+uPD0FoII/PkLxxzKV2JHu6YQtkLwBtz/DWR8bvGeseINcHg6PQp7e1hu1e282B/Pu5BvjDRjoY2LELgEnAOedo+l6KAPkD4ZeNtU+H3i2S0OnT3EN5Ktte2Cxfvy6sVXYDz5ilmG09ckHBwR6n8evh9fa8lp4k0eC7vLy3RbSazgj3kxbmYOoHzEhmwQAeGB4Ckn2yigD448HfFjxT4Js/sNhPBc2A3FLS8jLpGzEElSCGHQ8Z2/Mxxk5rsPE/7Q+qaxoc1hpGk/2RcTfK12t35roncJ8i7WPHzc45xg4I9/vvCfhvU7yS8v8Aw/pV3dSY3zT2UcjtgADLEZOAAPwrn0+D/gGOW1kHhyAta48sNLIQ2GLfOC2JOSfvZ4wOgAoA8z+B3ha40/xFbTavZ+XI1idTsD5oOUkCxq/ynurOMN0z0zivoSuS/wCavf8AcB/9r11tZUvtep3Y7/l3/gQVyXjv/mWv+w9a/wDs1dbWdrOhab4gs0tNUtvtECSCRV3smGAIzlSD0JqqkXKLSM8FWhRrxqT2Xbfb5Hgvxo+E01pcXni7QUkmtpXafUbbJZoWJy0q9yhOSw/h6/dzs5/4f/Gy+8E+H/7FuNKj1K0icta4n8logxLMpO1twLHI7jJ5IwB79/wrPwh/0CP/ACZl/wDi6P8AhWfhD/oEf+TMv/xdTer2X3/8A09ngf8An5P/AMAX/wAsPmnV9b8R/GXx5aWieWjzO0dlaGTEVtHjcxJ7nau5mxlsYA4VR7/a6VovwT+G99f29tJeywoj3Uwwsl1KWCLk87UDNwOdoJPzEktr/wDCs/CH/QI/8mZf/i6P+FZ+EP8AoEf+TMv/AMXRer2X3/8AAD2eB/5+T/8AAF/8sPlXx/40uPHnih9Yng+zRiJIYLfeH8pFGSNwVd2WLNyP4sdAK9I+DPxVvLGXSfBM2j/arWSUxW89rnzYt7O7M6nIZQWySNu1VJ+avYv+FZ+EP+gR/wCTMv8A8XR/wrPwh/0CP/JmX/4ui9Xsvv8A+AHs8D/z8n/4Av8A5YdbXzB+0d/yUPT/APsFR/8Ao2Wvcv8AhWfhD/oEf+TMv/xdH/Cs/CH/AECP/JmX/wCLovV7L7/+AHs8D/z8n/4Av/lh4L8MPir/AMK4ivNA1nR53tWu2lkaP5Z4JNoRlKNgHlFGCVI+brwBqfEH49Nr2lzaP4bspLazu7dobue9jXzSG4KoAxUDbkEnJ+bgKQCfZv8AhWfhD/oEf+TMv/xdH/Cs/CH/AECP/JmX/wCLovV7L7/+AHs8D/z8n/4Av/lh4R8OPAuqaZLoHiy/H2e3vNTtoLWB1+eVGbd5p/ur8gx/ezngYLfUlcxa/DzwtZXkF3b6XsngkWSNvtEp2spyDgtjqK6enCMrty6ixVWi6cKdFt8t9Wkt35NhRRRWhxHjWu+Drjxt4G8V2Fjzf2/iO4urVC4RZHUAFSSO6s2OnzbckDNfOtjfap4Z1yO6tZJ7DU7GUgErteJxkMrKfxBUjnkEV9hP4Bt/tl3cW+va9afap3uJI7W8Eab2OSQAv4fgKy9S+EGg6zcLcapqGrX06oEWS6mSVguScAshOMknHuawhzxjbl/E9XErC16rqe1te32X2OE8B/Gnxl4t8Y2GknQtNuLeR83JtUkRoYujSFmdgAuQcEc8KOWFc58bvhveaJrl34psIfM0i+l8yfZkm2mb727JPyuxJDdAW24Hy7varH4bWemWcdnYeIPENpax52QwXgjRckk4ULgZJJ/GrH/CCf8AU1+KP/Bj/wDY1XPP+X8TD2GF/wCf3/krPFfhZ8a/+Ecs49C8TmefTY9qWt2g3vbLkDaw6tGByMZZcYAIwF5f4m/FC8+IF5HbxwfZNItZWe2h3HfJkABpedpYYOMD5d7DJ6n3L/hRXg//AKe//IP/AMbqxY/Bjw3pl5HeWF1qdpdR52TQSRxuuQQcMEyMgkfjRzz/AJfxD2GF/wCf3/krOQ+H3w7uPAfibwvLqM27U9S+0vcQoQUgCQnagI+83znJzjsOm5vdK5iw8E29lrNpqkmsazfT2m/ylvboSqu5Sp/hz0PY9hXT06aleTkt3+iFjJ0nGnCnK/LG17W15pP9QrkviZ/yT3VP+2X/AKNSutrO13RrfxBo1xpd28qQT7dzREBhtYMMZBHUDtTqxcoOK6ojBVY0cVTqz2jJN+idz4l8N63N4c8S6brMHmF7O4SUokhjMig/Mm4dAy5U8Hgng1738Sfjb4duPCV3pXhub+0rrUYpLaR3hkjSCNl2sx3BSzEEgAcDqegDdH/worwf/wBPf/kH/wCN0f8ACivB/wD09/8AkH/43U88/wCX8TT2GF/5/f8AkrPKfgT4Yu5fE6+MLh47bRdKSfzLmR0CmTy8FTlgVAWTfuxj5cd+M/4xfEax8eapZwaZayLZ6a8yxXLtg3AfZ82zGVGUOMkkgjIU8V9BQfD2G1t4re38S+JIYIkCRxx3wVUUDAAAXAAHGKw/+FFeD/8Ap7/8g/8Axujnn/L+Iewwv/P7/wAlZ4z8GvHui+BtZvm1m0kKXyRxJexIHa3AY7gR12HIJ25P7sfKe3p/xj+K1jo1lqvhCxt5LjVLi38i4kdcRQJKhzznLPtYEY4G4Ek4K1r/APCivB//AE9/+Qf/AI3VzUvhBoOs3C3Gqahq19OqBFkupklYLknALITjJJx7mjnn/L+Iewwv/P7/AMlZ8s+FfENx4U8Uadrlqu+S0lDlMgeYhGHTJBxuUsM44zkc19D+LPj3pek6PYTaLp897dajafaYDcDy44gWkjO/BJZleMgqMAjo/StP/hRXg/8A6e//ACD/APG6sTfBjw3cWdtZzXWpyWtru+zwvJGUi3HLbVKYXJ5OOtHPP+X8Q9hhf+f3/krPlHSdSm0bWbHVLdY2nsriO4jWQEqWRgwBwQcZHqK+u/hx8SrP4iWd40Onz2V1ZbPtEbuHT5y+3awwTwnOVGM4561lf8KK8H/9Pf8A5B/+N1oaZ8J9H0Tzf7J1XWrDzseZ9kuEi34zjO1BnGT19TRzz/l/EPYYX/n9/wCSsqfGXwHceNvC8D6ZB52r2Eu+3TeF8xHIEiZZgo6K2Tn7mB1rwj4f+NtU+FfiieHUdOnS1uNiahZTReXMoAJR1DYIYByQDwwbtww+kv8AhBP+pr8Uf+DH/wCxrL1L4QaDrNwtxqmoatfTqgRZLqZJWC5JwCyE4ySce5o55/y/iHsML/z+/wDJWYmv/H/wva+GmutDkkvdWkQCKzmgdBExGcyNjBC9wrHJwAcHcPJPhr4LuPif4yvr/VZ82sMq3d++wfv3eTJjwrKV3gSfMv3cdORXtX/CivB//T3/AOQf/jdamm/C7TdGt2t9L1rXbGBnLtHa3SxKWwBkhUAzgAZ9hRzz/l/EPYYX/n9/5KzpfEegWPinw/eaLqSyG0ukCv5bbWUghlYH1DAHnI45BHFfGFst94J8a2r6nYSJeaTexTS2rttLFGD43cjBAGGGQQQRkV9bf8IJ/wBTX4o/8GP/ANjWPffBjw3qd5JeX91qd3dSY3zTyRyO2AAMsUycAAfhRzz/AJfxD2GF/wCf3/krMT/ho7wf/wBA3XP+/EP/AMdr5s1bUptZ1m+1S4WNZ724kuJFjBChnYsQMknGT6mvqb/hRXg//p7/APIP/wAbo/4UV4P/AOnv/wAg/wDxujnn/L+Iewwv/P7/AMlZgfADxxDqGkp4Naxkjn063kuEuRIGWVTMSwIwCpBkUDrnnp0PjHxB8DX3gXxLNYzxSGwldnsLgncJos8ZIAG8AgMMDB56EE/SWm/CDQdGuGuNL1DVrGdkKNJazJExXIOCVQHGQDj2FXL74bWep2clnf8AiDxDd2smN8M94JEbBBGVK4OCAfwo55/y/iHsML/z+/8AJWeU+Bv2g20zS49O8V2l3fGBCsd9bsrSuONqurEAnGfn3ZOBkE5Y8X8U/iV/wsPUbPyNP+x2Nh5og3vukk3kZZscLwi/KM4OfmPGPcv+FFeD/wDp7/8AIP8A8bo/4UV4P/6e/wDyD/8AG6Oef8v4h7DC/wDP7/yVnyrYX1xpmo21/ZyeXdWsqTQvtB2upBU4PBwQOtfUXwb8SXni6fX9cv44I7q6+z70gUhBtEiDAJJ6KO9WP+FFeD/+nv8A8g//ABuup8JeCdL8GR3SaW9wUuNm5ZSuF27sY2qMfeNJ88pRurW/yZrF4ejRqqNTmckktGvtRf6HSUUUVseYFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRXnPifx5qeiarrK/atDstN017dDLfrMXdpU3AKI8ljw3AHAGegJrI0z4qXOr+b9m8TeCY/Kxu+1tc22c5xjzQu7p2zjjPUVl7VdEzveXySXNOKuk7N9Grr8Geu0V5l/wAJ3qf/AENfw7/8GLf/ABVH/Cd6n/0Nfw7/APBi3/xVHtfJi+o/9PIfeem0VyX/ABcP/qV//Jij/i4f/Ur/APkxR7XyYfUf+nkPvOtorkv+Lh/9Sv8A+TFH/Fw/+pX/APJij2vkw+o/9PIfedbRXJf8XD/6lf8A8mK4vxx8T/EvgG4s7fVI9CnnukZ1jtBIzIoIGWDOpAJJAPfa3pR7XyYfUf8Ap5D7z2GivnP/AIaO1P8A6Btp/wB+G/8Ajtep6TqXj3WdGsdUt18NrBe28dxGsgnDBXUMAcEjOD6mj2vkw+o/9PIfedzRXJf8XD/6lf8A8mKP+Lh/9Sv/AOTFHtfJh9R/6eQ+862iuS/4uH/1K/8A5MUf8XD/AOpX/wDJij2vkw+o/wDTyH3nW0VyX/Fw/wDqV/8AyYo/4uH/ANSv/wCTFHtfJh9R/wCnkPvOtorkv+Lh/wDUr/8AkxR/xcP/AKlf/wAmKPa+TD6j/wBPIfedbRXJf8XD/wCpX/8AJij/AIuH/wBSv/5MUe18mH1H/p5D7zraK8+1/wAR+LPC2ltqWtX3hS0tA4TewuWLMegVVBLHqcAHgE9Aa5T/AIXb/wBTD4X/APALUP8A43R7XyYfUf8Ap5D7z2yivIrX4n6zqOjalqml3vhy/g07yvtCxW92jL5jbVx5gUHkHv2r12qjUUm1Yzr4WVGEZuSad1o76q1/zQUUUVZyhRXG2vxFt723S4tPDniOeB87ZIrEOrYODghsdQRUv/Cd/wDUqeKP/Bd/9lWXt6fc73lmLTs4fiv8zraK46f4hQ2tvLcXHhrxJDBEheSSSxCqigZJJLYAA5zWH/wvXwf/ANPf/kH/AOOUe2h3F/ZuK/l/Ff5nptFcFpnxY0fW/N/snStav/Jx5n2S3SXZnOM7XOM4PX0NaH/Cd/8AUqeKP/Bd/wDZUe2h3D+zcV/L+K/zOtorkv8AhO/+pU8Uf+C7/wCyo/4Tv/qVPFH/AILv/sqPbQ7h/ZuK/l/Ff5nW0VyX/Cd/9Sp4o/8ABd/9lR/wnf8A1Knij/wXf/ZUe2h3D+zcV/L+K/zOtorjbr4i29lbvcXfhzxHBAmN0ktiEVcnAyS2OpArsqqM4y2ZjXwtagk6kbX2+X/DoKKKKs5worLl8SaFBM8M2tadHLGxV0e6QMpHBBBPBpv/AAlPh7/oPaX/AOBkf+NTzx7m6w1Z6qD+5mtRWT/wlPh7/oPaX/4GR/40f8JT4e/6D2l/+Bkf+NHPHuH1Wv8AyP7ma1FZP/CU+Hv+g9pf/gZH/jR/wlPh7/oPaX/4GR/40c8e4fVa/wDI/uZrUVk/8JT4e/6D2l/+Bkf+NH/CU+Hv+g9pf/gZH/jRzx7h9Vr/AMj+5mtRWT/wlPh7/oPaX/4GR/40f8JT4e/6D2l/+Bkf+NHPHuH1Wv8AyP7ma1FZP/CU+Hv+g9pf/gZH/jR/wlPh7/oPaX/4GR/40c8e4fVa/wDI/uZrUVk/8JT4e/6D2l/+Bkf+NH/CU+Hv+g9pf/gZH/jRzx7h9Vr/AMj+5mtRWT/wlPh7/oPaX/4GR/40f8JT4e/6D2l/+Bkf+NHPHuH1Wv8AyP7ma1FZP/CU+Hv+g9pf/gZH/jR/wlPh7/oPaX/4GR/40c8e4fVa/wDI/uZrUVk/8JT4e/6D2l/+Bkf+NH/CU+Hv+g9pf/gZH/jRzx7h9Vr/AMj+5mtRWT/wlPh7/oPaX/4GR/40f8JT4e/6D2l/+Bkf+NHPHuH1Wv8AyP7ma1FZP/CU+Hv+g9pf/gZH/jR/wlPh7/oPaX/4GR/40c8e4fVa/wDI/uZrUVk/8JT4e/6D2l/+Bkf+NH/CU+Hv+g9pf/gZH/jRzx7h9Vr/AMj+5mtRWT/wlPh7/oPaX/4GR/40f8JT4e/6D2l/+Bkf+NHPHuH1Wv8AyP7ma1FV7PULLUYTNY3cF1ErbS8EgdQeuMg9eR+dWKpO+xjKLi7SVmcPpuq6dpnxC8W/b7+1tPM+x7PPmWPdiI5xk84yPzrov+Ep8Pf9B7S//AyP/GpbrQNGvbh7i70iwnnfG6SW2R2bAwMkjPQAVF/wi3h7/oA6X/4Bx/4VjGNSOit1/FnpVa2ErNSmpXtFaW+zFL9A/wCEp8Pf9B7S/wDwMj/xo/4Snw9/0HtL/wDAyP8Axo/4Rbw9/wBAHS//AADj/wAKP+EW8Pf9AHS//AOP/Cn+98jL/Yf7/wCAf8JT4e/6D2l/+Bkf+NH/AAlPh7/oPaX/AOBkf+NH/CLeHv8AoA6X/wCAcf8AhR/wi3h7/oA6X/4Bx/4UfvfIP9h/v/gH/CU+Hv8AoPaX/wCBkf8AjXkfxi+Jms2dxb6N4RuVME1uZbnULJhK3zFl8tWAIQgDdkHdyuCuOfXP+EW8Pf8AQB0v/wAA4/8ACj/hFvD3/QB0v/wDj/wo/e+Qf7D/AH/wPlvwj4a1Hxx4ltL/AMX60q6bbuFnl1a//eyIpDeUis4fDbj83AGWIJIwfqT/AISnw9/0HtL/APAyP/Gj/hFvD3/QB0v/AMA4/wDCj/hFvD3/AEAdL/8AAOP/AAo/e+Qf7D/f/AP+Ep8Pf9B7S/8AwMj/AMaw/C11b3vjrxdcWlxFPA/2PbJE4dWxEwOCOOoIrc/4Rbw9/wBAHS//AADj/wAKt2OladpnmfYLC1tPMxv8iFY92M4zgc4yfzpcs3JOVtP8i/bYWnSqRpKV5K2trfEn09C3RRRWx5x5t4r8W6R4M+Ip1TWZpI4DonlxrHGXaSTzWYIMcAkKeSQPUivlnXdT/tvxDqereT5P267lufK3btm9y23OBnGcZwK+09Z8H6D4gvEu9UsftE6RiNW850woJOMKwHUms7/hWfhD/oEf+TMv/wAXWKVSLdkvv/4B6c54StGDnOSaSWkU1p586/IyPCPxn8KeKntLNp5NO1S4cRLaXKnDPtBwsgG0gnKjJUkj7oyAfRK5L/hWfhD/AKBH/kzL/wDF0f8ACs/CH/QI/wDJmX/4uner2X3/APAMvZ4H/n5P/wAAX/yw62iuS/4Vn4Q/6BH/AJMy/wDxdH/Cs/CH/QI/8mZf/i6L1ey+/wD4AezwP/Pyf/gC/wDlh1tFcl/wrPwh/wBAj/yZl/8Ai6P+FZ+EP+gR/wCTMv8A8XRer2X3/wDAD2eB/wCfk/8AwBf/ACw62iuS/wCFZ+EP+gR/5My//F0f8Kz8If8AQI/8mZf/AIui9Xsvv/4AezwP/Pyf/gC/+WHW0VyX/Cs/CH/QI/8AJmX/AOLo/wCFZ+EP+gR/5My//F0Xq9l9/wDwA9ngf+fk/wDwBf8Ayw62iuS/4Vn4Q/6BH/kzL/8AF0f8Kz8If9Aj/wAmZf8A4ui9Xsvv/wCAHs8D/wA/J/8AgC/+WB/zV7/uA/8AteutrD0bwfoPh+8e70ux+zzvGY2bznfKkg4wzEdQK3KdOLSfN1IxlWnUlFUrtJJaqz08k3+YUUUVocgUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAHl2reCtN8eeKfFel6pPdwwRXFhcK1q6qxYQOuDuVhjDnt6V4L8SfAK/D3WbTThrEeoPcW/nnFu0TRjcVGRkgg7TjDZ4OQOCfp7w/wD8lC8Y/wDbl/6KNeQ/tLalDLrOgaWqyefb28tw7EDaVkZVUDnOcxNnjuOvbKj8Pzf5s7sx/jL/AAU//SInm/w88DzeP/EraTFfR2SR273EszRlyFBVcKuRk7mXqRxk9sH0DX/2ctX0/S2uNF1ePVrtXA+ytALcsp6lWaQjI4ODjjPOcA6f7Mv/ADNP/bp/7Wr6ArU4T5I+G/xe1TwN5Wm3KfbtCMu54T/rIAc7jEc4GSdxU8Eg42li1fW9fFHjma31v4la0+i2u+O51B0gS3kM/nuWwXQjr5jZYAf3sDjFfa9ABRUc88Nrby3FxLHDBEheSSRgqooGSSTwABzmuHn+M/w+triWB/EUZeNyjGO2mdSQccMqEMPcEg9qAO8rD8V+EtI8Z6MdL1mGSSAP5kbRyFGjk2socY4JAY8EEeoNaljf2ep2cd5YXcF3ayZ2TQSCRGwSDhhwcEEfhVigD4Q13TP7E8Q6npPned9hu5bbzdu3fscruxk4zjOMmvs/wJ/yTzw1/wBgq1/9FLXyB47/AOSh+Jf+wrdf+jWr7PB0vwzocMck8FhpljEkKPPNtSJBhFBdj9ByeaANCis/TNd0fW/N/snVbG/8nHmfZLhJdmc4ztJxnB6+hrQoA4v4o+NZvAfg46paQRzXktwlvbrKhaMMcsS+GU42o+MHrjtmuD+DvxX8QeLPEJ0LXPIuFW0klW6itmEjOHXG8p8irtJGdq8hRnJ59Y8T+GNL8XaHNpGrwebbycqy8PE46Oh7MMn8yCCCQcvwP8PNF8AW95FpL3cr3jq00t1IGYhQdqjaAABuY9M/Mck8YAOsooooAKKKKACiiigDn/GPg7S/HGh/2Tq3nrCsqzRyQPteNxkZGQQeCw5B6+uCPmz4v/DbTfh9caS2l3l3PBfJKGS62syMhXkMoAIIccY4weTnj6vE8LXD26yxmdEV3jDDcqsSFJHUAlWAPfafSvnz9pe+t5NR8O2CyZuoYp5pE2n5UcoFOenJjf8AL3FAGd8OdNhi+Cfi3VFaTz7i9gt3UkbQsbRspHGc5lbPPYdO/wBMV85/D3/k3rxJ/wBhUf8AtvX0ZWS/iv0X6ndU/wBxp/45/lTCiiitThOS+Gf/ACT3S/8Atr/6NeutrxbWp5rb9mK6eCWSJymwsjFSVa72sOOxUkEdwSK8t+CGm3eofFTTJLVpESzSW4nkQplY9hXGGByGZ1U4GcMSMYyMqH8KPojuzT/fq3+OX5s+r9W02HWdGvtLuGkWC9t5LeRoyAwV1KkjIIzg+hr48+JXgj/hAvFraVHdfabWWIXNs7DDiNmZQr8Y3AqRkcHg8ZwPs+vnj9pe+t5NR8O2CyZuoYp5pE2n5UcoFOenJjf8vcVqcJP+zL/zNP8A26f+1q+gK8r/AGfdNmsfhkLiVoyl/ey3EQUnIUBYsNx13RseM8EfQeqUAFFFFABRRRQByXxM/wCSe6p/2y/9GpXW1yXxM/5J7qn/AGy/9GpXW1kv4r9F+p3VP9xp/wCOf5UwooorU4ThPCOi6VqM3iSa+0yzupV1y6UPPArsB8pxkjpyfzrpf+EW8Pf9AHS//AOP/CsjwJ/zMv8A2Hrr/wBlrrawowi4K6PUzDEVo4mSjNpadX2Rk/8ACLeHv+gDpf8A4Bx/4Uf8It4e/wCgDpf/AIBx/wCFa1Fa8kexxfWq/wDO/vZk/wDCLeHv+gDpf/gHH/hR/wAIt4e/6AOl/wDgHH/hWtRRyR7B9ar/AM7+9mT/AMIt4e/6AOl/+Acf+FH/AAi3h7/oA6X/AOAcf+Fa1FHJHsH1qv8Azv72ZP8Awi3h7/oA6X/4Bx/4Uf8ACLeHv+gDpf8A4Bx/4VrV8sf8Ll8fX/jL7NpGpwTw3GoeXZ2fkRmORWkwke9kR9pBA3Ha2OTtPQ5I9g+tV/5397PpH/hFvD3/AEAdL/8AAOP/AAr5H8deF/EfgPXDYX93cS28mWtbtHYJcIO454YZGV7Z7ggn7Orx/wDaO/5J5p//AGFY/wD0VLRyR7B9ar/zv72ee/AKKPWfHV9b6oi30C6ZI6x3Q81Q3mxDIDZGcEjPua+jP+EW8Pf9AHS//AOP/CvCv2adNhl1nX9UZpPPt7eK3RQRtKyMzMTxnOYlxz3PXt9F0ckewfWq/wDO/vZk/wDCLeHv+gDpf/gHH/hR/wAIt4e/6AOl/wDgHH/hWtRRyR7B9ar/AM7+9mT/AMIt4e/6AOl/+Acf+FH/AAi3h7/oA6X/AOAcf+Fa1FHJHsH1qv8Azv72ZP8Awi3h7/oA6X/4Bx/4Uf8ACLeHv+gDpf8A4Bx/4VrUUckewfWq/wDO/vZk/wDCLeHv+gDpf/gHH/hR/wAIt4e/6AOl/wDgHH/hWtRRyR7B9ar/AM7+9nzL8d7q90rxBbaVZ6JbaRpYQSwXdrCqNeNj5suoBAUnGz6Mc5XB8GPFS6t4sh8Oa9p2n38FxblLWR9Ph3xvGpb5nABYFVbJbcSQvTJNd98YPijpGjaNq/hW1aSfWri3NvInkny4FkVclmJGSY3JXbuwRzisP4KfCrVNF1geKPENr9lkjiK2Nq5/eAuozIwB+X5Sy7W5yxyFKjJyR7B9ar/zv72exf8ACLeHv+gDpf8A4Bx/4Uf8It4e/wCgDpf/AIBx/wCFa1FHJHsH1qv/ADv72fJPiqx+IPhLxDqOrX+kwwQvnMtvYRz2CIX2rsDKyJkqANwD888sc8v4U1DV38W6TFaTQz3E13HDHFfgywOXYLiRTnK884GR1GCAa+vvHf8AyTzxL/2Crr/0U1fLHwfsbfUPivoEN1H5kayvMBuIw8cbyIePRlU++OeKOSPYPrVf+d/ez6X+H8UcEPiKGGNY4o9cuVREGFUDaAAB0FdhXJeBP+Zl/wCw9df+y11tRR+BHRmbvipt+X5IK8K+N2n65qHiWAaDaT3VzDYRySRwR+Y/l+a6kheSfmZegJ79Aa91rkv+avf9wH/2vSrJNJPuVl1SdOVScHZqL1W58gf8JBqn/P1/5DX/AAr6b8AeBNH1jwHo2o67o0Y1C5txI5S4kAdSTsfAfALJtYgYwSeB0Hzr8QRCvxG8SCCSR0/tO4JLoFO7zDuGATwGyAe4AOBnA+r/AIYyX0vwy8OtqMMcM4skVVQ5BiAxE3U8mMIT7k8DoH7Cl/KvuI/tTHf8/p/+BP8AzE/4Vn4Q/wCgR/5My/8AxdH/AArPwh/0CP8AyZl/+LrraKPYUv5V9wf2pjv+f0//AAJ/5nx/8RYte8M+MdQt2sJNN09rh1sR5ZaOWJcbWV23biVKlueC2MDoNT4Oo/inxydN1az/ALRsTaSSSfvfJ8jBXEny4LckJtB/jz2r2D486bDffCu8uJWkD2FxDcRBSMFi4iw3HTbIx4xyB9D5Z+zj/wAlD1D/ALBUn/o2Kj2FL+VfcH9qY7/n9P8A8Cf+Z7l/wrPwh/0CP/JmX/4uj/hWfhD/AKBH/kzL/wDF11tFHsKX8q+4P7Ux3/P6f/gT/wAzkv8AhWfhD/oEf+TMv/xdH/Cs/CH/AECP/JmX/wCLrraKPYUv5V9wf2pjv+f0/wDwJ/5nJf8ACs/CH/QI/wDJmX/4uj/hWfhD/oEf+TMv/wAXXW0Uewpfyr7g/tTHf8/p/wDgT/zOS/4Vn4Q/6BH/AJMy/wDxdYfjDwDpWmeE7++8PeGo77VIEV4bd552DjcN/AkBJ27iADkkADPQ+k0Uewpfyr7g/tTHf8/p/wDgT/zPhP8A4SDVP+fr/wAhr/hX0B8INE0Txv4Su9S1fSIPtEWoSwr5M0yDZtRwMbz03lR7AZyck8Z+0d/yUPT/APsFR/8Ao2WvQ/2dYJofhzdPLFIiTanK8TMpAdfLjXK+o3KwyO4I7Uewpfyr7g/tTHf8/p/+BP8AzOv/AOFZ+EP+gR/5My//ABdH/Cs/CH/QI/8AJmX/AOLrraKPYUv5V9wf2pjv+f0//An/AJnJf8Kz8If9Aj/yZl/+Lo/4Vn4Q/wCgR/5My/8AxddbRR7Cl/KvuD+1Md/z+n/4E/8AM5L/AIVn4Q/6BH/kzL/8XR/wrPwh/wBAj/yZl/8Ai662ij2FL+VfcH9qY7/n9P8A8Cf+ZyX/AArPwh/0CP8AyZl/+Lo/4Vn4Q/6BH/kzL/8AF11tFHsKX8q+4P7Ux3/P6f8A4E/8zkv+FZ+EP+gR/wCTMv8A8XR/wrPwh/0CP/JmX/4uutoo9hS/lX3B/amO/wCf0/8AwJ/5nJf8Kz8If9Aj/wAmZf8A4uj/AIVn4Q/6BH/kzL/8XXW0Uewpfyr7g/tTHf8AP6f/AIE/8zkv+FZ+EP8AoEf+TMv/AMXR/wAKz8If9Aj/AMmZf/i662ij2FL+VfcH9qY7/n9P/wACf+ZyX/Cs/CH/AECP/JmX/wCLrD8ReD9B8P3nh270ux+zzvrVtGzec75UknGGYjqBXpNcl47/AOZa/wCw9a/+zVnVpU1BtRX3HZgMwxdTERhOrJp30cnbZ+Z1tFFFdJ4pl+JJZIPC2rzQyNHLHZTMjocMpCEggjoa+UdR+KHiiHUJorbVtQSKM7MS3srsSBhjkEcEgkDHAIGTjJ+rPFP/ACKGtf8AXhP/AOi2ryv4keG9Fl+Bqa02l2g1SKzsXW8WILKT+6j+Zhyw2sRg5HT0GMJRUqln2PTo150cG5U3ZuXl2POfB3xC17WfGOlaXq2t6stne3C27NaXbiQM/wAqEbmIxuK54PGcc19B/wDCCf8AU1+KP/Bj/wDY18m+BP8Akofhr/sK2v8A6NWvt+q9jDsZf2liv5vwX+RyX/CCf9TX4o/8GP8A9jR/wgn/AFNfij/wY/8A2NdbRR7GHYP7SxX834L/ACOS/wCEE/6mvxR/4Mf/ALGj/hBP+pr8Uf8Agx/+xrraKPYw7B/aWK/m/Bf5HJf8IJ/1Nfij/wAGP/2NH/CCf9TX4o/8GP8A9jXW0Uexh2D+0sV/N+C/yOS/4QT/AKmvxR/4Mf8A7Gj/AIQT/qa/FH/gx/8Asa62vL7H4+eCr7XI9NDX0EckpiW+niVIO+GJ3blU8clRjPOBnB7GHYP7SxX834L/ACJ/GHh248P+Fr3VLTxP4jeeDZtWW/JU7nVTnAB6E969JrkviZ/yT3VP+2X/AKNSutpQio1Gl2X6muKrTrYSnOo7vmmvlaH+YUUUVseYFFFFABRRRQAUUUUAFFFFABRRRQAV4P8AFD4oeI/CXjK5sLC6xajbsTy4/l/doTyUJOSxr3ivNpfB+geMPHXiqDXtNjvEgezeIl2RkJhIOGUg4PGRnBwPQVlVV3Fef6M9DAy5IVqlk2o6XSf24rZprZs8c/4X14v/AOfn/wAhxf8AxuvR/hr4l8X/ABD06+uv7fn077JKseW02KWOTIz8r4X5h3XHAZTnnj558S6bDo3irV9Lt2kaCyvZreNpCCxVHKgnAAzgegr6L/Zx/wCSeah/2FZP/RUVHsY+f3v/ADJ/tGt2h/4Lh/8AInZ/8I/4v/6Hj/ykxf40f8I/4v8A+h4/8pMX+NdbRR7GPn97/wAw/tGt2h/4Lh/8icl/wj/i/wD6Hj/ykxf40f8ACP8Ai/8A6Hj/AMpMX+NWfH3iWbwh4I1PXbe3juJ7ZEEcchIXc7qgJxyQC2ccZxjIzmvlCH4neN4NUOop4n1Izl2fY8xeLLZz+6bKY54G3A4xjAo9jHz+9/5h/aNbtD/wXD/5E+pv+Ef8X/8AQ8f+UmL/ABo/4R/xf/0PH/lJi/xrZ8N6u3iDw1pusPZyWZvbdJ/Id1cqGGRyvBBHI6HBGQDkDUo9jHz+9/5h/aNbtD/wXD/5E5L/AIR/xf8A9Dx/5SYv8aP+Ef8AF/8A0PH/AJSYv8a62ij2MfP73/mH9o1u0P8AwXD/AORPGviP4q8R/Duzs2m8UT3t1e7/ALPGmmQInyFN25iSRw/GFOcY4615x/wvrxf/AM/P/kOL/wCN19IeJfB+geMLeCDXtNjvEgcvES7IyEjBwykHB4yM4OB6CvizXdM/sTxDqek+d532G7ltvN27d+xyu7GTjOM4yaPYx8/vf+Yf2jW7Q/8ABcP/AJE+lvh5rPinx/4abVovFUlk8dw9vLC2mwuAwCtlW4yNrL1A5yO2T1n/AAj/AIv/AOh4/wDKTF/jWB8BtNhsfhXZ3ETSF7+4muJQxGAwcxYXjptjU855J+g9Mo9jHz+9/wCYf2jW7Q/8Fw/+ROS/4R/xf/0PH/lJi/xo/wCEf8X/APQ8f+UmL/Gutoo9jHz+9/5h/aNbtD/wXD/5E5L/AIR/xf8A9Dx/5SYv8am8E3+pXtnqseqXv2yez1Oa0Wbylj3KgX+FRjqSe/WunrkvAn/My/8AYeuv/ZanlUZqzfXqzb28q+GqOcY6ctrRinv3SR1tFFFbnlhRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAHJeH/+SheMf+3L/wBFGvDf2jv+Sh6f/wBgqP8A9Gy17l4f/wCSheMf+3L/ANFGvDf2jv8Akoen/wDYKj/9Gy1lR+H5v82d2Y/xl/gp/wDpETn/AIVan410jUdVvPB+l/2lstM3cMgZowoOVO0Mu6ThgoGWIL4B5rQ8f+LPiZ4j0d11/Rr7S9GTYZo4tPlghZg3yl2fJPJXgttyFOMjNdf+zL/zNP8A26f+1q+gK1OE+WPgNrPhbRvFF5Jr7wW188QFheXJCxxcN5g3E4VmG3BPYMMjdhvqevkj44aBZ+H/AIivHYWcFna3VpFcJDBgIOqHCBVCcoePmz1z820ehw6rfQfsmm6S5kM5t2td7/OfKa6MRT5s8eWdo9BjGMCgDiPGPijWvjF48t/DujyRjTRcOljEXKI4UHNxJkA52hjjGVHABJO70+D9njwgujRWlxc6lJeBw8l7HKqM3y4KhCCoTPzdCw6biK8A8DeFda8XeJY7HQZo7e8gQ3X2h5TGIAhGHyuWzuKgbQTkg9ASPe/+Ee+Of/Q56H/35X/5HoA8gP2j4P8AxcFvDqc88NhLCLqWGIIbiB1R5E2FiDwcDJ6gHIIBH13BPDdW8VxbyxzQSoHjkjYMrqRkEEcEEc5r501v4GfEDxHrE+ratrWh3F9Pt8yXzJE3bVCjhYQBwAOBXvfhrTZtG8K6Rpdw0bT2VlDbyNGSVLIgUkZAOMj0FAHxh40hW28deIYEMhSPU7lFMkjOxAlYcsxJY+5JJ717Xo3wI1TXtmoeP/Ed9PdGIIkENx5skQ4YBpZNw4JcFVBGeQxrxTxpPDdeOvENxbyxzQS6ncvHJGwZXUysQQRwQRzmvuOgD5E8d+FL74R+NbGfRtSuyGQ3NneGLYUO5lMZPKuQu3dwAQ/KgHB+o/Cuvf8ACT+F9O1r7FPZfbIhJ5E4+ZeccHupxlW4ypBwM4r54/aO/wCSh6f/ANgqP/0bLXr/AMEv+SQ6F/28f+lElAHz58T/AIeal4F1SCa/1WPU01J5HS5IYSuy7S5kBzgln67mz1OOlex/s4/8k81D/sKyf+ioq5/9pr/mVv8At7/9o10n7Osap8ObplmjkL6nKzKobMZ8uMbWyAM4APGRhhznIABwnxs8CeJNPln8R3Otz6rpDXbmOGQyE2Pmsx2hSWVYxhF3ZGSVG3pXF+BE8aa8l94R8MXsiWl8gkvYTKqIse5UZyTyB8yhgnLDghgMV7v+0FqU1j8MjbxLGUv72K3lLA5CgNLleeu6NRzngn6jl/2ZoJlt/Etw0Uggd7ZEkKnazKJSwB6EgMpI7bh60AYE/wCzl4mh0aW4TU9Nn1BHJW0jLBXQLniRgPnLcbSAO5YdKsfADxzfW3iBPCF1LJNp92kj2iEZ8iVQXODnhGUOSOfmwRjLE+9+LL640zwbrl/ZyeXdWun3E0L7QdrrGxU4PBwQOtfLHwS/5K9oX/bx/wCk8lAH1fret6d4c0efVtWuPs9jBt8yXYz7dzBRwoJPJA4FfMniXxp4j+MXjGDw9oryWmm3DmK3s3m2K6r85knxwxATdjnbtwoJyW9L/aO/5J5p/wD2FY//AEVLXOfszQQtceJbhoozOiWyJIVG5VYylgD1AJVSR32j0oArw/s06k2lmSfxJaJqGxiIEtmaLdztHmEg4PGTs4yeDjnyvxpo/iDQ/FF1aeJnnm1Lgm5mlaXz0xhXV25ZcDA9MYIBBA+36+f/ANpr/mVv+3v/ANo0AU/h7/yb14k/7Co/9t6+jK+c/h7/AMm9eJP+wqP/AG3r6MrJfxX6L9Tuqf7jT/xz/KmFFFFanCfHfi7xN4p/sC00CYz23hw5NuohKJdMrF2Jcj59rPggHaNq8ZGawPCMHiObxLaP4Viu31aFw8TWy5Kchct2CfMAS3y4ODwa9v8AFNjb3f7MqzTx75LSUTQHcRsc3RjJ46/K7Dn19cVT/Zl/5mn/ALdP/a1ZUP4UfRHdmn+/Vv8AHL82dx4W1n4rXfiO0g8S+GdKstIbf9ongkUumEYrgCZurbR0PX8a4f8Aaa/5lb/t7/8AaNfQFfP/AO01/wAyt/29/wDtGtThOM+HnjXx+ult4P8AB8EdxK7vcRyMm97dflLBS7eWiEj+IdXOOWFbnifwZ8XPC+hzavN4svr23g5mWy1W5d407uQwX5R3xnHXGASPR/gDa2dv8L4Jba482a5u5pbpN4bypAQgXA+78iI2Dz82ehFdh47/AOSeeJf+wVdf+imoA5f4Q/Ej/hOdDe21KaAa7Z8TInymePjEwXGBknBC5APPyhlFegX99b6Zp1zf3knl2trE80z7SdqKCWOBycAHpXyx8AUuH+KEBgs4J40tJjPJKAWt0wAHTJ4YsVTjPyu3GMkdR+0j4huG1HSfDSrttUi+3yHIPmOS8a9sjaFfvzv6cCgCPUvjR408Zayul+A9KkthvBVliWedl3Fdz7gUjQ7kzkfKR9/Brn7yb4weALi11rVZ9Za3hfc32i8N3bkZC7ZQrsFB3ADODk/KcjIk+EvxV0fwDp17Yalo88v2mUzG8tNjSHAULGVbb8o+c53cFjxyTXX+K/jr4P8AEPhLVtIGlaq8l3aSRRefbwlFkKnYx/eHGG2nIGRjI5oAvn4mWPj/AOFOqo4jttat0hN1aA8Eecg8yPPJQnt1UnBzkFvaa+F/DU80OsokUsiJMjJKqsQHXG7Deo3Kpwe4B7V90Vkv4r9F+p3VP9xp/wCOf5UwooorU4Ty5fGdj4F8NeKtYvo5JifEFzDbwJwZpSAQucYUYUkk9ADgE4B8N1LXfG/xh8QLZRpJdOEEiafasY7eIICC5DNgH5j8zHOWCg9BXR/Fua+XRpoIxJ/Z7+Jr95yI8r5qrGI8tjg7WlwM889ccbP7NFjbyaj4iv2jzdQxQQxvuPyo5csMdOTGn5e5rKj8CO7Mv96l8vyRyer6N8Q/gy9pLDqskVnO7MsllK0lqZSuCro6hd+0AjcvIGQcqcc/qviDxp8Sri3t7v7XrU9kjvFHa2Slo1YqGJESA4yEGT7etfSfxt/5JDrv/bv/AOlEdeefszQQtceJbhoozOiWyJIVG5VYylgD1AJVSR32j0rU4Tv/AIN+Fdf8J+DpLPxBNIsr3DPDZGVZFtk9iOhZtzEBiOQeCWzxd/8AGXxT4q8UXOh/DvSYLmHynEdzPGfMOAczDcyoi5I2hwcnGeW2j2DxVoP/AAk/hfUdF+2z2X2yIx+fAfmXnPI7qcYZeMqSMjOa4P4N/DzWvAT6+usPaMLp4Fha3kLBwisS3IBAzJjnByrcYwSAcJrPxG+L/gnUXu/ENhALW4yIoprZHtoyxJCrJEc7gFYBWcnHJB4Nez+BfHWl+PNDF/YHyriPC3Vo7Ze3c9j6qcHDd8diCBx/xz8VaFB4E1Tw8+pwHV7jytlohLuNssbndjOz5eRuxntmuX/Zl/5mn/t0/wDa1AHcfEr4tWfgKVtMjsJ7nV5bQXFsWAEA3Myjed27jaTgDngZGcj5U0nUptG1mx1S3WNp7K4juI1kBKlkYMAcEHGR6ivufVprG20a+n1QRnT47eR7oSR71MQUl8rg7htzxg5r5I+DWlLq3xU0ZJbaSeC3d7p9u7EZRCyOxHQCTZ14JIHOcUAe16F8fvDet6jpmm/2ZqsN9fSxW+NkbRpI5C/e3glQT12g47dqr/tHf8k80/8A7Csf/oqWvYK8v+P2p/YPhfPbeT5n9oXcNtu3Y8vBMu7GOf8AVYxx97PbBAPHPhB8SdN+H1xqy6pZ3c8F8kRV7XazIyFuCrEAghzznjA4OeOo1L40/E3RrdbjVPCVpYwM4RZLrTrmJS2CcAs4GcAnHsar/s6eHtN1PWdZ1S9to7ifT0gFssqK6ozszbxkEhwYhggjGT68fSdAHn/wv+Jtv8QNOmjuI4LPV7XHm2qSE+YmF/eqCOFLEjGW28ZPzDPoFfHngGeG1+NOmPbyx6TAdTeONY2F2qKxZRCHGQ4YHy/M/wBrd2r2/wCPviGbRvh8LK1uY4p9UuBbuu8rIYQCzlcEHGQit1GHweooAw/HX7QNvpd4bDwnBBfzQylZ7u5UtAwA6R7WBbkn5uB8vG4EEchd/GD4o+Hv7O/tmx+z7fN/4/8ATWh+2f733fuZGNm3tuzXP/CzxX4T8I6jeX/iTR57+6/dGxeKFJfIKklmw7ABshMMMkYOCMnPvc/xm+G11by29xrcc0EqFJI5LCdldSMEEGPBBHGKAI/hv8XtL8c+Vptyn2HXRFueE/6ucjO4xHOTgDcVPIBONwUtWf4/+Nf/AAg/ih9F/wCEanutkSSefLceQsm4Z+QbG3KOm7j5gwxxk+AaJ4ks/DnxKg1/SY57fTINQaSOHaHkW1ZiCnzEgt5ZK8nr3719P/F7Tv7R+F+tBdPgvZreIXEazceVtILSKcjDKm8jnnphgSpAOH0f9pHS7q8ePV9CnsLcRO6yw3H2gs4GQm3auN3IBz1IzgZIwLH9pHWP7cja/wBGsRpBlO+OAP56xnOMMW2sw4/hAbGPlzked/Dbwuvi/wAeabpU8cjWe8zXe1GI8pBuIYggqGICbsjBcd8Cvf7/AOAHgu51SzurWO7s7eJwZ7NJ2dJ1G443MS6kkrkhuikAAncAD0C88N6LqGs2usXml2lxqFqmyC4liDNGNwYYz3DDIPVcnGNxzqUUUAY/ifxPpfhHQ5tX1efyrePhVXl5XPREHdjg/kSSACR4h4h/aRvGvAvhrRoEtV6yakCzycD+FGATB3fxNng8dK4v4x+M77xR41urCeOOGz0e4mtLaJeSSG2u7NjJLFBx0AAHXJPvfwn+H6eCfC8JvrWAa7cbnupgFd4wxGIg4GdoCqSMkbtxBIxQBwetfHrTdT8K69omqeH9S07VJ7e4s1hUrIqMyFPnLbGUhiQRtOMfgPPPgl/yV7Qv+3j/ANJ5K+g/i94XsfEfw+1Oa4jjW7023kvLW4KZaMoNzKORwyrtPbocEqK+fPgl/wAle0L/ALeP/SeSgD6S8Cf8zL/2Hrr/ANlrra5LwJ/zMv8A2Hrr/wBlrrayo/AjuzL/AHqXy/JBXJf81e/7gP8A7Xrra5L/AJq9/wBwH/2vRV6eoYH/AJef4H+h82/G3/kr2u/9u/8A6Tx16PbfHnQvDPg3w/punWM+q31vp9tFcDcYI4mWPay7ipJYFR0XaQfvcYrzj42/8le13/t3/wDSeOva/g14N0WL4eaNqd5oFodUld7r7Rc2waUHzD5bqWGVG1UI24HcdcnU4TD0z9pTR5fN/tbw/fWuMeX9kmS43dc53bMdumc5PTHPtFhfW+p6dbX9nJ5lrdRJNC+0jcjAFTg8jII615/8Yfh/b+L/AAvcX9ra7td0+IvbPGDvlQHLREAEvkbto/vEYIBbPnn7PXjaaHVJfCF9cSPb3CNLYKxJEci5Z0XjgMuW5IAKHAy9AHpfxt/5JDrv/bv/AOlEdeQfs4/8lD1D/sFSf+jYq9r+LU19B8K/ED6cJDObcI2yPefKZ1WXjB48svk9hk8YzXhH7PupQ2PxNFvKshe/spbeIqBgMCsuW56bY2HGeSPqAD6rrxPX/wBo3SNP1RrfRdIk1a0VAftTTm3DMeoVWjJwOBk45zxjBPSfHOz1S8+F96NNOY4pY5b2MJuZ4FOTjg4w2xyeMKjc44NP4C+G77w/4Ku31TS5LG8u71nHnxbJWiVVVdwPzABvMwD6kjhskAw9D/aNs9S1yysb/wAP/YLW4lWJ7s34cQ7uAzAoo2g4ycjAyecYPuFfOn7Rfh7SNOuNG1SytrS1vLx5xcrEhVrjBVt5wNpILHJJDHeOoHy+h/Aqa4k+FGmpPa+THFLOkD+YG85PMYl8D7vzFlwf7mehFAHpFeR6/wDtC+FNNRl0eG71ifYGUqhgizuwVZnG4EDnhCOQM9cc3+0D4/8A+ZL06T+7JqZaL/deJFY/gxwP7oz94VB8H/g/o2v+HF8R+I1+2w3m9LW0SR4xGFcqXYqQSxKkAA4A5OSflAOz039oLwRfXDRXB1LT0CFhLdWwZScj5R5bOc856Y4PPTPqleJ/GD4beFNM+HlzrGmaVHY3mnJBHE9uSodPMVMOOjHDk7j8xIGWI4OJ+zn4n1SXVr3wxLP5umR2j3cKPyYXEiAhT2U7ySPUZGMtkAj/AGltNhi1nQNUVpPPuLeW3dSRtCxsrKRxnOZWzz2HTv6X8Ev+SQ6F/wBvH/pRJXn/AO01/wAyt/29/wDtGrngL4n+HPB/wUsjPdx3Gp2jzRDTkbEryNK7rweibWBL8gcjlvloA6jW/jp4Q0PVNQ06ZNSnuLG4+zuILdcOwzvKlmHCsu05xkkbdwyRqeF/ix4W8W6jqVnYzzw/YImuHmu4xHG8KnDSBs8KMqfn2nDDjg48g8L/AA/uPi/4o1Lxdqtr/YuhXUrMiWgAeZ8Y+QkYOCMvIR8zbgBknb7Hofwp8H+Hre5istNkL3dk1jdSyXMhaeJgN4PzYBbAOVAx2xQBl/8AC9fAP9o/Zv7Un8nyvM+1/ZJPL3Zxsxt37sc/d24754rc8QfEnwp4b0u21C81WOaC7SVrQ2gM4uDHgMqsuVzkgckDPfg4+aPiz4K03wH4qtdL0ue7mglskuGa6dWYMXdcDaqjGEHb1roPhT8HYfG2lz6zrVzd2un7zFbJbgK8rDG59zKRsHK8A5IbkbeQD0/Tf2gvBF9cNFcHUtPQIWEt1bBlJyPlHls5zznpjg89M+qV8wfFr4S6P4E8PWWraTf30vm3YtpIrso+dyMwYFVXGNhGCDnI6Y56f9nDxO89nqfhi5n3fZ8Xdoh3EhCcSAH7oUMUIHHLseecAHvFcH4r+L/hDwlcG0uLyS+vFfbJbWCrK0XLA7iSFBBXBXO4ZHGOa5P47/ES40Gz/wCEU0+HbcalaF7m5cAhYHLIUUf3m2sCT0HTk5XiPhV8Gv8AhLbO38Q63ceXpBlPlWsf37pVJDZYH92u4Y/vHDfd4YgHp+jfHzwVq+opZytfabvwFmvolWMsSAAWRm29c5bCgA5Ir1CvnT4sfCHw54V8Jprek3slo9qkcDW07eYb2QsBuBJG19u9iFBBC8BcE1J+zn4n1SXVr3wxLP5umR2j3cKPyYXEiAhT2U7ySPUZGMtkA9j8Z+OdF8C6Wl9rEshMr7IbeABpZjxnaCQMAHJJIA4HUgHzub9pHw2t5bLDo2qvatu+0SOI1ePj5dqhiHyeuWXHXnpXkHxc8Q3HiH4las867I7CVrCBMg7UiYg8gDOW3NznG7GSAK9b8PfC74X+PPCsF3oK6lCInWOa5WZlnMgQFkcOGTPzgkoMZ6HHFAG54O+OPh/xXrn9kS20+mXE0rJZtOylJxxsBI+7I3Py8jjAYkgV1njPxzovgXS0vtYlkJlfZDbwANLMeM7QSBgA5JJAHA6kA+CfCPwN4p0j4r2FxqOg31rb2P2jz55oisY/dug2uflfLMMbScjkcAmuL+J2pTar8TfEVxOsaul69uAgIG2I+Up5J52oCffPTpQB7Ppv7Segy27Nqmh6lbT7yFS1ZJ1K4HJZihBznjHYc88Enxa0fx54h0PSdNsL6DydTtbkS3IQbsOysuFY4xuQg55y3TA3R6N8LvAPjX4ape+HbXytTltAizteyO0N0qglZQfl+9w2EGVOVAypry/4b6TqWnfEawS+0+7tXgu7ZJlnhZDGzyKyBsjgsoJAPUAkVlW+Bndlv+9R+f5M+wKKKK1OEyfFP/Ioa1/14T/+i2rj/Femf2v8AJ7bzvK2aLDc7tu7Pkoku3GR12Yz2znnpXYeKf8AkUNa/wCvCf8A9FtXN6wJm+BV0IJI0f8A4R0kl0LDb5HzDAI5K5APYkHBxg5f8vfkd3/MD/2/+h8teBP+Sh+Gv+wra/8Ao1a+36+IPAn/ACUPw1/2FbX/ANGrX1f8SvG//CBeEm1WO1+03UsotrZGOEEjKzBn5ztAUnA5PA4zkanCWPGPxA8P+B7PzdWut1w20x2UBVp5ASRuCEjC8N8xIHGM5wDxcH7RXg2a4iie01mBHcK0slvGVQE/eO2QnA68An0BryT4X+FH+JnjmaXxBc313a20Qnu5nZnacgqqRNITlcjPfO1CBjqPa9V+BHge90u4t7DTpNPu3TEV0lxLIYm7Ha7kMOxHoTgg4IAO40DxHpHinS11LRb6O7tC5TeoKlWHUMrAFT0OCBwQehFalfHmka34j+DXjy7tH8t3hdY720EmYrmPG5SD2O1tytjK5wRyyn2v9oa+uLT4axwwSbI7vUIoZxtB3oFeQDnp8yKePT0zQBY1n4+eCtI1F7OJr7UtmQ01jErRhgSCAzsu7pnK5UgjBNXIPjj8PpreKV9akgd0DNFJZzFkJH3TtQjI6cEj0JrxT4NeFPB3izWJ7XxDczvfp81tp+7yo7hNp3HeDuZh12grgLn5huC2Pib8Hb/w9rkcvhiwvtR0y83OkNvBJM9qRjKMQDleflJOTgg5xuIB7/8AEe+t9P8Ahr4jmupPLjbT5oQdpOXkUxoOPVmUe2eeK+NNJs4dR1mxsri7js4Li4jikuZMbYVZgC5yQMAHPUdOor6D+Kvw71TxD4N0rXvOg0+60fSs3mlA4togse9xCF3BWBG3HIIC8jbz8+aVpV9rmqW+maZbSXN5cPsiiTqx/kABkkngAEnAFAH198TL+z/4Q3VNO+1wfbvKin+zeYPM8vz0Xft67c8Z6Zrta+W7j4KeJPCMDa5f3ulSWtrjekEshc7jsGAYwOrDvX0poVreWPh7TLPUbj7RfQWkUVxNvL+ZIqAM248nJBOTyayX8V+i/U7qn+40/wDHP8qZoUUUVqcJyT+Prf7Zd29voOvXf2Wd7eSS1sxIm9TggEN+P4ise++M/hvTLySzv7XU7S6jxvhnjjjdcgEZUvkZBB/GuQ8R/Ej/AIQbQ/Edtps0B128166EKP8AMYI+MzFcYOCMANgE8/MFYV5J4L8Da78SNcn+zy5jWVX1C/nkDGPfuO4gnc7Ha3TqepAOawhzyjfm/A9XEvC0Krp+yva32n2PoL/hevg//p7/APIP/wAcrYsfiTZ6nZx3lh4f8Q3drJnZNBZiRGwSDhg2Dggj8K8p1X9my+tdLuJ9M8Qx315Gm6K2e08kSn+7v8wgHGcZGM4yQORw/wALfiDc+B/EsAnnkbRbh/Lu4GkfZGGKgzKgyN6hR2JIBXjIIrkn/N+Bh7fC/wDPn/yZn0t/wnf/AFKnij/wXf8A2VH/AAnf/UqeKP8AwXf/AGVeE/G3wZ4rtvEt94kv5JNR0d3VYblcYtUJO2JkzlQp43YwxYEncxFcH4Tv/FMeotpPhW7vorvU9sLRWchRpNp3A5GNuME7sjClskKWo5J/zfgHt8L/AM+f/JmfTV98Z/DemXklnf2up2l1HjfDPHHG65AIypfIyCD+NbH/AAnf/UqeKP8AwXf/AGVfOr/AzxzFod1qUthAklvk/YVnEk8qjGSgTKnqeN247TgE4zH8HPGd94X8a2thBHHNZ6xcQ2lzE3BBLbUdWxkFS546EEjrgg5J/wA34B7fC/8APn/yZn1L4f8AEFv4js57i3t7q38idreSO6QI6uoBIIBOOuPzrWrkvAn/ADMv/Yeuv/Za62nSk5QTZGOpQpYiUIbf8AK5Lw//AMlC8Y/9uX/oo11tcl4f/wCSheMf+3L/ANFGlU+KPr+jLwn8Gv8A4F/6XA+TfHf/ACUPxL/2Fbr/ANGtXv8A8LfHPhbw58IdJXVtesbeaDzvMt/NDzLuuHx+6XLngg8Dpz0rwT4gwTW3xG8SJPFJE51O4cK6lSVaQsp57FSCD3BBrrPhP8J08fRXWpaleT2umW0ohAgC75n2ksAxJ2bcxnlTndgYxxqcJ7vY/GDwDqF5Haw+I4EkfODPFJCgwCeXdQo6dzz0613FfKHxd+F9n4A/sy60me+uLG73xyG5UN5Ui4I+dQB8wJwpGfkY5Pb0/wDZ78T3GseEr3SLyeeebSpUETSYISB1+RAepwUfr0BUA4GAAeqXs+ms8el38toXv0kRLSdlzcKF+cBD98BTyMHg815n4R8DfCrTvFC3ej6rY6lqTSmWztm1KOfyCAx/dopy2BzltxGwEEEZrlPjv8NpvtF/46tLyMwFIheW8uQwbKRKY8DBBGMg4xgnJzgef/BiNpfi3oKpNJCQ8rbkCkkCFyV+YEYIGD3wTgg4IAPsOsPxL4w0DwfbwT69qUdmk7lIgUZ2cgZOFUE4HGTjAyPUUeMPEsPg/wAJ3+vT28lwloikQoQC7MwRRk9BuYZPOBng9K+SNC0XX/ij41dGnkuby4cT3t05XMUW5UZ9pKghQwwi9gAAAKAPpP8A4Xb8PP8AoYf/ACSuP/jdSeKPi94Q8OaW9xDq1pql2UYwWtjOspkYY4ZlyEHIOW7A4DEYrzjxH+z5baR4IvLvTbvUtV16BA6RxqiRyjeN2I8FshM8BiSRx1xXiehaZ/bfiHTNJ87yft13Fbebt3bN7hd2MjOM5xkUAfY/gjx7o/j7Trm80kTxfZpfKkhuQiyDIBDbVZvlPIBPUq3pXy58WtNh0r4qeILeBpGR7gXBLkE7pUWVhwBxucge2OvWvpf4c/Dmx+Hml3MEF1JeXl24a5uWXYHC52KqZIUAMe5JJPOMAfOnxt/5K9rv/bv/AOk8dAHufwn1Wx0P4G6Xqep3MdtZ26XDyyv0UfaJPxJJwABySQBkmtjTfi14D1W4aC38S2iOqFybpXt1xkDhpFUE89M56+hr5w8CfDTxH8QLdTazx2+i29wyvPNLlUkIj37Iwclyuw9gdoBYYFU/HXw41j4f/YP7WubGb7d5nl/ZHdsbNuc7lX++Ome9AH2fRXz/APs5+LLyWW98JTJ5lrFE99bylyTF8yK0YB42ktuGMYO7ru49c8feJZvCHgjU9dt7eO4ntkQRxyEhdzuqAnHJALZxxnGMjOaAOkrkvAn/ADMv/Yeuv/Za+aYPi94vXxjFr9xq13JGLgSyafHOyW7R9DGEOVA28ZwSD82S3NfS3gT/AJmX/sPXX/stZT+OPzO7D/7rW/7d/M62iiitThCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAOK02/s9M8beNLy/u4LS1j+w75p5BGi5jIGWPAySB+NeC/HjXNL17x9by6TfwXsMGnxwySwPvTfvkbAYcNw69CfTqCK6D4teHYde8damx1i0sJ4HhCJdRylZFaFdxDRqxBBVeCOdx5454L/AIV7/wBTTof/AHzd/wDxiuenVpxTTkt3182ezjMBiqs4zp0pNOENUm18EfI9M/Zl/wCZp/7dP/a1e6alq2m6NbrcapqFpYwM4RZLqZYlLYJwCxAzgE49jXyHB4EmtbiK4t/F2jQzxOHjkj+2KyMDkEEQZBB5zUdn4XvvEd4b/wAQeJILWaSVUnkvTPcTsgAG8bVYNgcAFx93HAwav29L+Zfecn9l47/nzP8A8Bf+RU+IXiz/AITXxpfazGk8dq+2O2hmfcY41AA9lycsVHALHk9T9B6X4Gvrn9ndfC93FINQksnnjhQ7GEpkM8cbbwNp3bFYHGPmGe9cv8P9D+G/hSK2vdW1ODU9bhlE8dwLW4CQHaBtVSNrbW3EOVB5BwCBXp//AAszwh/0F/8AyWl/+Io9vS/mX3h/ZeO/58z/APAX/kfMvwn8Y2/gnxzDf33FhcRNa3ThC7RoxBDAA9mVc9fl3YBOK+v7G/s9Ts47ywu4Lu1kzsmgkEiNgkHDDg4II/Cvnf4g+FPAviK9m1bw94hjsNQmdpLiG4t7hop5GfcX3bSUPLcAEHCgBeSfL4fCV7LqhtHubSGAOy/bXZzEQM4bCqXwccfLnkZA5we3pfzL7w/svHf8+Z/+Av8AyPoz4s/FmHwbbvo+jvHN4glTk4DLZqRwzDoXI5VT/vHjAb0Dw2NXXw1po16SN9W+zobsogUeZjkYBIyOhI4JBIABAHhfw58OeAfC1xb6xruuR3+sQuzRRx28xt4TkbWAMYLOME5OACeBlQ1WPitcQ+Nb2BtF8bR2+ntbiK50+4W6jidlcsr7VjYMTkdQMbF5PY9vS/mX3h/ZeO/58z/8Bf8AkeSaTqWpaz8S7HVLdbRdUvdYjuI1kDCATPMGAOCW2bj6k4r7br4z/wCFe/8AU06H/wB83f8A8Yr2Dwx4oudP8A6xo2t+OILzU5opo9Pukjui8JZDgtMUDHDnI+XK46ngKe3pfzL7w/svHf8APmf/AIC/8jjP2jv+Sh6f/wBgqP8A9Gy17P8AB+xuNP8AhRoEN1H5cjRPMBuBykkjyIePVWU+2eea+a9b8KajcefqVz4nsdXvm27h5ly00vRR80sSjgerDgcdhXoHwlu5fBX20X3imxisby0LpZiC4m8m6O3azLsUcDcG2P8ANgcnAIPb0v5l94f2Xjv+fM//AAF/5Gp+0vDbtp3h2ZrrbdJLOkdv5ZPmIQhZ93QbSqDHff7Gtj9nH/knmof9hWT/ANFRV43feDLzU7yS8v8AxnpV3dSY3zTteSO2AAMsYMnAAH4VufD7VfEPgXxZDYQ6zaN4fluFe9fa7wOu35iilBIr44BCjLKu7Kij29L+ZfeH9l47/nzP/wABf+R6h+0NNbx/DWNJ7XzpJdQiSB/MK+S+1yXwPvfKGXB/v56gVz/7NF9cSad4isGkzawywTRptHyu4cMc9eRGn5e5rh/GOkXfjLxjquq3HirTTAbhks/tK3Axb9UCqsJCgBsEcEsGPOdxr6BoGr+FtUXUtF8baNaXYQpvVLpgynqGVrchh0OCDyAeoFHt6X8y+8P7Lx3/AD5n/wCAv/I+mPHf/JPPEv8A2Crr/wBFNXzB8Ev+SvaF/wBvH/pPJXf/ABW8S3viy9g0fQvEVonh28twt4k9u6hJUcuGY+UXwcIBszypyAOT4/pVp4i0PVLfU9M8y2vLd98UqSJlT+eCCMgg8EEg5Bo9vS/mX3h/ZeO/58z/APAX/kfRn7QWmzX3wyNxE0YSwvYriUMTkqQ0WF467pFPOOAfofNP2dtb+w+ObvSZLjZDqVodkWzPmTRncvOOMIZT1AP1xXs0HxF8LXWjRW+s6laTTy24S8jjtJmgdiuHADJkoTkYPbrXzp468I6Taa4ZfB979t0yfLiF1dHtT/cJcDcvPByT2PTcx7el/MvvD+y8d/z5n/4C/wDI+xK+ZP2jdVa68a6fpi3MckFlZBzEu0mKWRju3Y5BKrEcHtgjrzz3/CZ/E7+x/wCy/wC3L77P/f8AOTzvvbv9d/rOv+1046cVzepeEr2xuFit7m01BCgYy2rOqg5PynzFQ54z0xyOeuD29L+ZfeH9l47/AJ8z/wDAX/keqfD3/k3rxJ/2FR/7b19GV8U6Da63pty0W+eCxnx9piSfCS7QSu5QfmwTkZHFfa1TCUZVJOLvov1NcVQq0MHTjVi4vmnumukO4UUUVueYfOfxC/5N68N/9hU/+3FXP2Zf+Zp/7dP/AGtWvd+A28d/BPSYLVo01Cye4uLYmJWaU7pR5IYkbA7bMnOPlGRxx4bps3ivwL4xa208Xdhr0bm1MCRh2ctgBNmCHBypHBB+Ujsayofwo+iO7NP9+rf45fmz7br5s/aTksT4q0aKOGQagtkWnlJ+Voi58tRz1DCUngfeHJ7el+BPFHxA13TtXv8AX/DkFp9liIsrPyJLaS7mwTjdI52rwoyVwS/X5SK8I8Y2vxE8ca5/a2reFNVWZYlhjjg0uZUjQZOBkEnkseSevpgDU4T3P4DW0MHwrs5IrWSF7i4mklkZgROwcrvUZOBtVVwQvKE45ydD4zzzW3wk154JZInKRIWRipKtMisOOxUkEdwSK84+D3ivxfoes2HgXU/Dt2bMpJMgktmhnto2bJkO7AMW7d15y2ASQEL/AI46341vrzUNAsdEvk8NW8SPc3Udm0iXGAspYybcKqEY4P8AC2SegAMD9nH/AJKHqH/YKk/9GxVY/aRsbiPxlpN+0eLWbT/JjfcPmdJHLDHXgSJ+fsa4v4YeIde8NeKnvfD+iyaxO1uY57aOF5G8nehYjZypyqjcQQM9DX0P4o0SH4xfDK2ks/temvK4vLIXsYQllDqvmAZ+RlYkEHoytz0IBn/ACDSF+G6XFhFGL97iRNQkCnc0isSgJPUCNkIA4G49y1d5feE/Dep3kl5f+H9Ku7qTG+aeyjkdsAAZYjJwAB+FfKGg+MfGPwp1i80uP9w0cubrTrxN8bPtwG4ORkEHcjDcAvJGK9Avv2l7ySzkWw8MQQXRxsknvDKi8jOVCKTxn+IevPSgDe+JN54B0Mt4c0/RNNi8QTorrJZWMSm2AIb53GCpZQ2AMnB5wCCfaa+R9G8NeI/E1pqfxC1i4kMCOCJ5h813IzCMhBwAig4yOBtCgcHb9cVkv4r9F+p3VP8Acaf+Of5UwooorU4TyqXwppfjDwl4usNWufscMOvXVzHdlsC3dVH7xskAqAWyD2J5BwR4j8KfFWveFvEs82i6Nd60k1uRc2FuHJZQRtf5Q2CrEDJU8Mw43Zr0f4h/2j/wq/xZ9i/49/8AhKJPt33f9Tlcdef9Z5X3efwzXEfC74rQ/D63ubK40SO7gu7hJJLmFwk6KBgjkEOAOVXK4Jbn5uMqPwI7sy/3qXy/JHZ/ES+8Y/FKKLQtG8DarZ2NtKLvz9Sj+zvIQu0D5yEXBduAzE8HjBFekfDL4d2/w+0OSIzfaNTvNr3syk7CVztRAf4V3NyRk5JOOFHL/wDDR3g//oG65/34h/8AjtdZ4S+KnhTxncRWenXskWoSIziyuoykmFPODyjHHzYVicZPY41OE0PHPjOx8C+GpNYvo5JiXENvAnBmlIJC5xhRhSST0AOATgHwDQND8ZfHG/v7zUtdktdJiuAzowkaBHKNtWGLOwlRgHLAgOCSxPPqfxx8Hap4r8JW8ukefPcadKZmskfidCuCQmPmkXjbz0LgZJArxD4a/FO8+Hn26D7D/aNjd7X8hrgxeXIONynDDkcHjJ2rzxggHofxI+FngvwZ8MLy8tIJDqiPGlvdXV03mSM0oyAoIQnZv4C9FJ7E1H+zL/zNP/bp/wC1q5zx94i8X/FPw/Hq9r4Yu7Lw7paNNIyytIsrElTJyFDhApHyqdmXJIB44vwX4/13wHeTz6PJAY7jb9ogniDpLtDBckYYY3k/KR75HFAH1v47/wCSeeJf+wVdf+imr5g+CX/JXtC/7eP/AEnkr6v13TP7b8PanpPneT9utJbbzdu7ZvQruxkZxnOMivjjRNQ1T4afECC7vNOxf6ZKyzWk525DKVYAj1ViVYZHIPI6gH2vXj/7R3/JPNP/AOwrH/6KlrU8MfGbTfGXiyx0TRdF1JklSV7m5uAqC3VVypwpbILYXJK4LL1zivLPjT8ULPxds8PaTBusbK7aSS7ZgfOkTeg8vaSDHgk7jycjgY5AN/8AZl/5mn/t0/8Aa1fQFfJnwg+JOm/D641ZdUs7ueC+SIq9rtZkZC3BViAQQ55zxgcHPHufxN+KFn8P7OO3jg+16vdRM9tDuGyPBADS87gpycYHzbGGR1AB8yfD6ZoPiN4bdBGSdTt0+eNXGGkCnhgRnB4PUHBGCAa9b/aa/wCZW/7e/wD2jXhmk6lNo2s2OqW6xtPZXEdxGsgJUsjBgDgg4yPUV9f+K9D0j4qfD4pZXUc8c6fadOuUkKqswDBS3BOMkqykZGW4DAYAPLPgX4J8I+J/D11favp8F7qdnqHCvM/yx7EKbow20qWD/eBBwRzjFepx/CXwHFpc2nL4atDBK+9nZnaUHj7spbeo+UcBgOvqc/Mngbxpqnw38US3HkTtGN0N9pzv5XmEAgBsqdrK3PTI5HAJr2uf9pDwqtvK1vpWsyThCY0kjiRWbHALByQM98HHoaAOoudJ+Gnw+TS1vNP0bT3e4C2c1xCJJfM3bt3mMGcBSR85OFyvI4q58VP7R/4Vf4h/sv8A4+Psh3/d/wBTked97j/V7/f05xXiHhOG8+MvxhbWtVtsaZZ7Z5YCTJHHGnEUOWBU7m5YHaGHmkAdK6v4sfGHRX0vXPCNhp8l7eF2s5pbmICCMj7zLzuLqwwOAAw3AkAZAOI/Z91KGx+Jot5VkL39lLbxFQMBgVly3PTbGw4zyR9R9V18UfD3xTb+DPGljrd1Y/bIYdyMqsQ8YYFS6cgFgCeG4OSODhh9f+FvEln4u8OWmuWEc8drdb9iTqA42uyHIBI6qe9AGxRRRQB8EX9jcaZqNzYXkfl3VrK8MybgdrqSGGRwcEHpX3nBPDdW8VxbyxzQSoHjkjYMrqRkEEcEEc5r5Q+OHhKbw748uNRSGNNP1d2uICshYmTC+cGB5B3tu9MOMdCB2Hwm+M+maR4fTQPFM8lulkmLS92yS70zxGwAYgrnCkDG0Y42jcAex+O/+SeeJf8AsFXX/opq+YPgl/yV7Qv+3j/0nkr0f4nfG3Qr7wve6J4Zmnuri+i8qS78kxxxxsSHXDjcWKjH3QAHyGyMV4h4V17/AIRjxRp2tfYoL37HKJPInHytxjg9mGcq3OGAODjFAH114E/5mX/sPXX/ALLXW1w3wu1KHWdG1fVLdZFgvdWmuI1kADBXVGAOCRnB9TXc1lR+BHdmX+9S+X5IK5L/AJq9/wBwH/2vXW1yX/NXv+4D/wC16KvT1DA/8vP8D/Q+bfjb/wAle13/ALd//SeOvp/wJ/yTzw1/2CrX/wBFLXxx4svrfU/GWuX9nJ5lrdahcTQvtI3I0jFTg8jII619L/DP4meFrrwNpVle6xY6dfWFpHbTQXc4j+4CisGcKG3BA2Fzt3AH31OE7TxoYV8C+IWuI5JIBplyZEjcIzL5TZAYggHHfBx6Gvlz4Jf8le0L/t4/9J5K7j45/E6z1OzTwt4fvoLu1kxJqE8IEiNgq0aI+cHBG5seijP3hV/9nzwKkVm3jO9G6abzILFCqkKgOHlB5IYkMg6YAbqGGAD0j4qReb8L/EK/2j9gxaFvO3bd2CD5fUf6zHl4778YPQ+Gfs6wQzfEa6eWKN3h0yV4mZQSjeZGuV9DtZhkdiR3rv8A47+OdFtvDF/4QEsk2rXaROUiAKwBZEceYc8FlBIAyehOAQT5h8E/GGmeD/FV/PrepSWenz2RTASR1eUOhXKoDyF34JHGT60AfRfjrx1pfgPQzf35824kytraI2HuHHYeijIy3bPckA+OeHfEvxc+KL6i+j6naaXp4Ty3cQiKJWKgFI32vJvwd2Qflz1XKiq/7RtnDLrPh7Xre7jngvrJoo/LwylUYOHDA4YMJv06nPFj4J/Ejwv4T8K3+l67fSWc7XpuI2MDyK6siLgbASCChzkDqMZ5wAcf8VvBeveFLjR7jXvEEmtT3tuy+ZK7u0TIQWQFySUBkBB4zk/KO/vfwS/5JDoX/bx/6USV4R8YfiJb+PNctotOh26ZpvmJbzOCHnL7dzkH7q/IMDGe567V6z4S/F+HSrXw/wCCp9GkZHuDbi9S4BO6WVip8sqONzgH5umTz0oA4f4zxtF8W9eV5pJiXibc4UEAwoQvygDABwO+AMknJP1H4E/5J54a/wCwVa/+ilryf9onwdcXcFp4utfnjtIha3is4GxC/wC7ZRjn5nYHnuuBjcaxPhz8dYfDXh+30PxBYXdzBaIywXVs4eTbkbUZXIGACwBDcAKNvGaAPpOub8NeAfC/hC4nuNC0mO1nnQJJIZHkbaDnALsSBnBIGM4Gegrj9f8Aj54P03S2n0eeTWLzeFW2WOSAY7szumAAPQE5I4xkin8MvGni7xDLqHirxVd2OneFVi8qAOqQReduRdys3zFRhgSzY3PgZwQoBwH7R3/JQ9P/AOwVH/6Nlry+70TUbHR9O1a5t9ljqXm/ZJd6nzPLba/AORgnHIGe1dx8bvE+l+KPHwm0if7Rb2doto06/ckdXdiUP8S/OBnvg4yME+j/AAxttC+IXwXl8H3C/wCl6f5gd5YifIkkeR4pkIIzjJGMgnawPynkA7T4VfECHx34aBl8watYJHFfBlGHYg4kUgAYbaxwMbSCMYwT3lfFH/E9+F3xA/5YJq2lS+0kbqy/+gsjezAN/CRx9V+DPiN4c8dI66RcyLdxJvls7hNkqLuIzjJDDpypONy5wTigDwz9o7/koen/APYKj/8ARstev/BL/kkOhf8Abx/6USV4p8fdW03WfHVjcaXqFpfQLpkaNJazLKobzZTglSRnBBx7ivS/2fNZ0KXwW2jWb+Xq8MslxewuTmXccLIuTyoUIpxjBHI+YFgA/aO/5J5p/wD2FY//AEVLXIfs16Z5viHXNW87H2a0S28rb97zX3bs54x5OMY53dsc7/7RmuaXJ4ZstEjv4JNTTUEmktUfc8aCJ+WA+7/rExnGc5GcGuc/Zvu7GDxLq8E+oyQ3lxbottZl8JcAEs7YxguoAxzkBn4IyQAcn8bf+Sva7/27/wDpPHWh4M+EfjDX/DkeuaJrFjZWuoxSwOjXM0bvGHKMjhUIKkp0yQeK6D9onwj9j1i08V2y/ub7Ftd89JlX5G5b+JFxgAAeXk8tV/4H/E7R9M0NPC2uX32SSOWWS1nnCRwLGcNsL54YsZG+bjnGc4FAGZefAf4g6hb2tve+ItNuYLRNltHNe3DrCuAMIDHhRhQMD0HpW38Pfgf4g8L+NLHW9R1exjhs9zhbItI8hIK7DvQAKQxyeT2GCdw9k1/xHpHhbS21LWr6O0tA4TewLFmPQKqglj1OADwCegNeN/DzWvFnj34r3fiWC6vofCtvLIDbyTukJHllI0Ee4qZOVdscA5ORlQQDY+Kvwa/4S28uPEOiXHl6uYh5trJ9y6ZQAuGJ/dttGP7pwv3eWPgH/FU/DzxD/wAv2janF9U8xQ//AHzJGWT3Vsd69Q0T4t6ppHxh1dPEWpzw6FLdzwzW9yvmfZBHvEYRY9wVgQqnbkNkk5OGHsfxF0Tw5q/g7UG8R/ZIIILd/Lv5o9zWjHGGTGGzuCfKD8+AvOcUAcf8MfjUvjLVI9C1ixjtNUkRmhlt9xinK7mK7TkoQozySDhuRwDn/F74PX3ibVJfEnh5o5L+RI0nsWO0zEfL5iuzbQQuwbcAYUnOeD5h8DoJpvi3pDxRSOkKTvKyqSEXyXXLeg3Moye5A710nhT4z6j/AMLVuLjUNSnfw7ql2YhDdbQLWMkiFh8wWPbld5BwRuJ3HBoA8r03Vda8J6y09hc3em6hA5jkC5RgVYEo6nqNyjKsMccivZ/D/wAWrzx5qOj6Tq1hBBfQ6zbXMctoCI2jB2lSGYkMCwOQcEE9MfN3nxs8PaNqvw/vtT1NvJutMiL2dxh22OzINm1SAd5VUyQdu7Pavnv4V/8AJQ9J/wCvmH/0alZVvgZ3Zb/vUfn+TPsyiiitThMnxT/yKGtf9eE//otq5/Uv+SHXf/Ytv/6TGug8U/8AIoa1/wBeE/8A6Laub1eOxm+B8kWozWkFu+hovm3YzGjmIeWxwCch9pGATnGATisv+XvyO7/mB/7f/Q+WvAn/ACUPw1/2FbX/ANGrXr/7TX/Mrf8Ab3/7RrxzwXPDa+OvD1xcSxwwRanbPJJIwVUUSqSSTwABzmvoP9onRPt3ga01aO33zabdjfLvx5cMg2txnnLiIdCR9M1qcJn/ALNemeV4e1zVvOz9pu0tvK2/d8pN27Oec+djGONvfPHuFfOH7OHiL7Nrmp+HppcR3kQubcPNgCROGVUPVmVskjnEXcDj6Hvr+z0yzkvL+7gtLWPG+aeQRouSAMseBkkD8aAPlj4/WKWnxQnmSOdGu7SGZzKylXIBjymOQuEA+bncG7Yr2seG7H4m/BzRLO/v5JZXsoJkvUk8xkuVj2szc/OQxdWBOeW5B5HgniqbVPix8UNRfw9az3ytlLVPM+VYIxjfl9ojVj82DjDSY5J59b+MHjW++HejaDoXhYyWLlMLMYvMVII1CLGDIrBicgk53DaM/fBoA8E8T+CfEHhG8mh1fTZ4oY5fKW7WNjBKSMjZJjByATjrwcgEED0z4ffHq+014dM8WtJfWbOqLqH/AC1t024+cAZlGQCT977x+c4Fe1/D/wAY2/jjwlbatF8twuIbyMIVEc4UFwuScryCOTwRnnIHyh8RbHRtM+IGsWGgR+XptrKIUTc52uqqJBl+Thw/9OMUAfW/jv8A5J54l/7BV1/6Kavlj4P/AGz/AIWvoH2HyPO8193n52+X5b+ZjH8Wzdt7bsZ4zXvemSX0v7N7tqMMcM48OTqqocgxCFhE3U8mMIT7k8DoPmzwK9vF4+8PzXd5BZ28OoQSyTzkhFCuG5IBxnGMngZ5IGSAD6y+Jn/JPdU/7Zf+jUrra4b4o6tpsPhG/wBLl1C0TUJkieK0aZRK6+avKpnJHytyB2PpXc1kv4r9F+p3VP8Acaf+Of5UwooorU4T4z+Kn/JQ9W/6+Zv/AEa9e+fAHTPsHwvgufO8z+0Lua527ceXgiLbnPP+qznj72O2T4H8VP8Akoerf9fM3/o169o/Z68Vw6j4Tl8NymNLvS3Z4lGAZIZGLZ65JVywJAAAZO5rKj8CO7Mv96l8vyR7JXw540ghtfHXiG3t4o4YItTuUjjjUKqKJWAAA4AA4xX23f31vpmnXN/eSeXa2sTzTPtJ2ooJY4HJwAelfFH/ACOvxD/58v7b1X/rp5PnS/huxu9s47VqcJ9P/G3/AJJDrv8A27/+lEdeIfAHTPt/xQgufO8v+z7Sa527c+ZkCLbnPH+tznn7uO+R7v8AGC1+2fCjX4vtEEG2JJd877VOyRH2g/3m27VHdiB3rxj9nH/koeof9gqT/wBGxUAfT9fCGhXVnY+IdMvNRt/tFjBdxS3EOwP5kauCy7TwcgEYPBr7nv8A7H/Z1z/aPkfYfKf7R9ox5fl4O7fnjbjOc8Yr4U0mOxm1mxi1SaSDT3uI1upYxlkiLDew4PIXJ6H6GgD7G8Cf8zL/ANh66/8AZa62uS8Cf8zL/wBh66/9lrrayo/AjuzL/epfL8kFcl4f/wCSheMf+3L/ANFGutrkvD//ACULxj/25f8Aoo0VPij6/owwn8Gv/gX/AKXA+bfjb/yV7Xf+3f8A9J469f8A2cf+Seah/wBhWT/0VFXhnxO1KbVfib4iuJ1jV0vXtwEBA2xHylPJPO1AT756dK9z/Zx/5J5qH/YVk/8ARUVanCZf7S2pTRaNoGlqsfkXFxLcOxB3Bo1VVA5xjErZ47Dp3wP2bNSmi8VazparH5FxZC4diDuDRuFUDnGMStnjsOnfp/2k9Shi8K6NpbLJ59xem4RgBtCxoVYHnOcyrjjsenfjP2cf+Sh6h/2CpP8A0bFQB7/47/5J54l/7BV1/wCimr5U+Esl9F8VPD7adDHNObgqyucARFGErdRyIy5HuBweh+q/Hf8AyTzxL/2Crr/0U1fJnwy1uHw98SND1G48vyFuPKkaSQRrGsimMuWPACh934dR1oA+j/jboDa98Mr5olkafTnW+RVZVBCAhy2eoEbOcDByB16HxT4B6zZ6R8SlivH8v+0LR7OFyQFEhZHUEkjrsKjGSWZRjmvqPStVsdc0u31PTLmO5s7hN8UqdGH8wQcgg8ggg4Ir5Ih8Kal8TfiN4gbw4I5LSS9uLk3k+5IkjeRihY4yC3ZcZ68YBIAPpP4p6jeaV8M9bvbDUPsF1HEoS4GcrudVIUgEhmBKg8YLA5XGR8wfC3RrzW/iVoUVmmfs13HeTOQdqRxMHYkgHGcBRnjcyjIzXpknwT+IOuPDaeI/Gkc+nh953XdxdFGCkArG4UE84zkYBP0Pq/gb4faL4F0uOCxgjmvyhFxqDxgSzE4JGeqplRhAcDAzk5JAOsr4o+I99cah8SvEc11J5ki6hNCDtAwkbGNBx6Kqj3xzzX2vXxB47/5KH4l/7Ct1/wCjWoA+u/h9BDbfDnw2kEUcSHTLdyqKFBZowzHjuWJJPckmvI/2l7//AJF3Tku/+e88tssn+4qOy/8AfwAn/ax3r1T4Y6bDpXwy8O28DSMj2SXBLkE7pR5rDgDjc5A9sdeteV/tNf8AMrf9vf8A7RoAwP2cf+Sh6h/2CpP/AEbFX0P4p8N2fi7w5d6HfyTx2t1s3vAwDja6uMEgjqo7V8yfAbUprH4qWdvEsZS/t5reUsDkKEMuV567o1HOeCfqPp/xHr9j4W8P3mtak0gtLVAz+Wu5mJIVVA9SxA5wOeSBzQB4Jof7Oetx65ZSa3f6VJpiSq9zHBLKzyIOSg+VcbumdwxnPOMV7F4E/wCZl/7D11/7LXnmjftI6Xd6ikOr6FPp9q+B9ohuPtGwkgZZdqnaBkkjJ44BzXofgT/mZf8AsPXX/stZT+OPzO7D/wC61v8At38zraKKK1OEKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigDkv+FZ+EP+gR/5My//ABdH/Cs/CH/QI/8AJmX/AOLrraKy9hS/lX3Hd/amO/5/T/8AAn/mcl/wrPwh/wBAj/yZl/8Ai6P+FZ+EP+gR/wCTMv8A8XXW0Uewpfyr7g/tTHf8/p/+BP8AzOS/4Vn4Q/6BH/kzL/8AF0f8Kz8If9Aj/wAmZf8A4uutoo9hS/lX3B/amO/5/T/8Cf8Amcl/wrPwh/0CP/JmX/4uj/hWfhD/AKBH/kzL/wDF11tFHsKX8q+4P7Ux3/P6f/gT/wAzkv8AhWfhD/oEf+TMv/xdH/Cs/CH/AECP/JmX/wCLrraKPYUv5V9wf2pjv+f0/wDwJ/5nJf8ACs/CH/QI/wDJmX/4uutooq4wjH4VYwrYqvXt7ablba7b/MKKKKowPmPxrqesa54hu/h3pMO3+0demuZJQz/PlyoVgo/1a7S7Eg9AeNvPok37PPgqSztoEl1WKSLdvnS4UvNk5G4FCox0G0L75PNeR+P01TRfiHL4rt7Pfa2usyCKZxmMzxymTY2DkZBB7ZGcHg49k8I/HTwt4j22+pP/AGJfHPyXcgMLfePEuABwB94LywAzWVH4Ed2Zf71L5fkjP/4Zx8H/APQS1z/v/D/8arxz4mfDOb4dXGnK2qR6hBfJIUcQmJlZCu4FcsMYdcHPrwMc/S+pfE7wRpVus9x4n010ZwgFrMLhs4J5WPcQOOuMdPUV4h4t1TV/jv4ltLPwvo8iafpiHNxdMEKeaVDNIQSAPl4VdzEKx56LqcJ7n8OvENx4q+H+j6xeLi6miKTHI+d0Zo2fgADcVLYA4zjtWxdaFo99qMGo3mlWNxfQbfJuZrdHkj2ncu1iMjBJIx0NSaTpsOjaNY6XbtI0Flbx28bSEFiqKFBOABnA9BVygArP/sLR/wC2P7X/ALKsf7T/AOf37Onnfd2/fxu+7x16cVoUUAFc/wCJPA/hrxd5Z1zSILuSPAWbLRyADOF3oQ235iducZOcZroKKAMfw94V0LwpZm10PTILKNvvlAS8mCSNznLNjccZJxnA4qNvBvhl3vXl0DTZTfXAurnzrZXEkoUqHIYEZwW6d2Y9WYncooA5/wD4QTwf/wBCpof/AILof/ia0NT0LR9b8r+1tKsb/wAnPl/a7dJdmcZxuBxnA6egrQooA5//AIQTwf8A9Cpof/guh/8Aia3IIIbW3it7eKOGCJAkccahVRQMAADgADjFSUUAZet+G9F8R2/kazpdpfIEdEM0QZowww2xuqE4HKkHgelcPH8BvAaapNdtY3ckDptWya7fyozx8ykYfPB6sR8x46Y9MooAp6bpOm6Nbtb6Xp9pYwM5do7WFYlLYAyQoAzgAZ9hVO+8J+G9TvJLy/8AD+lXd1JjfNPZRyO2AAMsRk4AA/CtiigDj9T+FfgbV/K+0+GbGPys7fsim2znGc+UV3dO+cc46muosbCz0yzjs7C0gtLWPOyGCMRouSScKOBkkn8asUUAFFFFAFe+sLPU7OSzv7SC7tZMb4Z4xIjYIIyp4OCAfwryPX/2dfDmoO0ui6hd6S7OD5bD7REqhcEKGIfJODkue/HTHslFAHB+Gvg/4N8NW88Q0yPVHmcMZdUjjnZQBwq/KAo6ngZOeScDGX/woHwN/bH237PffZ/+fD7UfJ+7jrjzOvzff6+3FeoUUAcV8NrG30zTtcsLOPy7W11meGFNxO1FCBRk8nAA612tcl4E/wCZl/7D11/7LXW1lR+BHdmX+9S+X5IK43WYNesvHSa1peif2lAdMFow+1JDtbzS/wDFz0A7d67KiqnDmVr2McNiHQk5cqldWad7Wfo0/wATwuL4V2o1G+vLn4cT3X2uUyrC2uxxR2+SSVjWIJhecYOcBRjvnL1/4NSakjNo/g270efYFULrMU8Wd2SzK/zEkccOBwDjrn6IoqfZy/mf4f5G31uj/wA+IffP/wCTPnvw78IJ9G1GO8v/AAZPq3lxR4hudVhWPzgcs+1RypwAEbIA3Z3ZGPV/+Eg8X/8AQj/+VaL/AArraKPZy/mf4f5B9bo/8+IffP8A+TPJtf8ACY8Uao2p6x8Lo57xkCNKuuCIuB03bCATjjJ5wAOgFchP8FUmuJZU8J6zAjuWWKPXbYqgJ+6N0ROB05JPqTX0RRR7OX8z/D/IPrdH/nxD75//ACZ59MmsXPhoeHZ/h1HLpIt1thbPq8bARqAFGTzkYGGzkEA5zzXnn/CmYv7R+0/8Ibqvk+b5n2T+3rfy9uc7M+Xv244+9ux3zzX0JRR7OX8z/D/IPrdH/nxD75//ACZ51rFrqevaGmi6l8NYJtNj2eXAuqxRiLbwuwqAUwOPlxwSOhIrlPC/w1PhbVE1KHwHd393E6vA99rUDiFhnlVVFBPIOWBwVBGCK9woo9nL+Z/h/kH1uj/z4h98/wD5M5L/AISDxf8A9CP/AOVaL/CvMPE/wofxBeTXtr4Hn0i4eLYsdlqlssAcDAcx7PpkKVzjsSSffKKPZy/mf4f5B9bo/wDPiH3z/wDkz578MfCCfRZYbnVPBk+s3UfJWbVYY4CwbKnywMnAABDMynJyOcD0PxH/AMJD4p8P3mi6l4GkNpdIFfy9YiVlIIZWBx1DAHnI45BHFeg0Uezl/M/w/wAg+t0f+fEPvn/8mfPdj8GYrS8jnm8G6rexrnME+vW4R8gjkpGrcdeCOnpxWx4I8Cav4D8Q3Orab4Wvp/OtPswiudXtzty4ZmyqDOdqADHGG65G32yij2cv5n+H+QfW6P8Az4h98/8A5M8i8b+FL/x79mk1XwLPDdW/CXVprECSFOfkJKEFcnPI4OcEZOczwl8OLnwd4gi1qy8HalcXcKMsX2rXLcqhYbSwCxrk7SRzkcnjOCPcKKPZy/mf4f5B9bo/8+IffP8A+TPnP/hSH/Us65/4PbT/AOM1seFvhpeeEfEdprlh4V1WS6td+xJ9btSh3IyHIEQPRj3r3Sij2cv5n+H+QfW6P/PiH3z/APkz5j134MeL9b8Q6nq32Dyft13Lc+V50TbN7ltud4zjOM4FZ/8AwoXxf/z7f+RIv/jlfVlFHs5fzP8AD/IPrdH/AJ8Q++f/AMmcFrcmveI9Hn0nVvh/9osZ9vmRf2zGm7awYcrgjkA8GvGL74EeI5LyRrDT54LU42Rz3UErrwM5YMoPOf4R6c9a+pKKPZy/mf4f5B9bo/8APiH3z/8Akz5T/wCFC+L/APn2/wDIkX/xyva/Dn/CQ+FvD9noum+BpBaWqFU8zWImZiSWZicdSxJ4wOeABxXoNFHs5fzP8P8AIPrdH/nxD75//JnzHrvwY8X634h1PVvsHk/bruW58rzom2b3Lbc7xnGcZwKz/wDhQvi//n2/8iRf/HK+rKKPZy/mf4f5B9bo/wDPiH3z/wDkzybwZ4YvfAqO2kfD6RruVNkt5caxC8rruJxnaAo6cKBnauckZrzjVvgj4p1HWb69t9Ljs4Li4kljto5IdsKsxIQYcDABx0HToK+oKKPZy/mf4f5B9bo/8+IffP8A+TPl+8+DfxB1C3tbe9nu7mC0TZbRzXSOsK4AwgMuFGFAwPQeldH4N+FGq+H9R0u5k0Kc3sd7DJcXr38RRYlckhYlORkbSSSx+TjGSK99opOk2rOT/D/IuGPhTlzQoxT7+/8ArMKKKK2POM7X7Wa98OapaW6b557SWONcgbmZCAMnjqa+fPE3wf8AFPiDVI7+LS5LZ/ssEEqtdQyBmijWPcvzLtBVF4Oec884H0pRWcqd3zJ2+7/I66OLVOm6Uqakr315t9ukkfKf/ChfF/8Az7f+RIv/AI5Xp/gjS/Hfhfw9c6Hq2hf29YyfLGlzfRjy49gQxYZnBjwBhQABluuePXaKXs5fzP8AD/Iv63R/58Q++f8A8mfL+q/AzxDdapcT6ZpEljZyPuitnu4pjEP7u/eCRnOMjOMZJPJk1P4RfEfW/K/ta8vr/wAnPl/a7xJdmcZxulOM4HT0FfTlFHs5fzP8P8g+t0f+fEPvn/8AJnj3gHwnqvgC3kaz8ESXWoToqz3s2qwbiABlUAHyIWGccnpknAxU+JvhLxf8Rv7L/wCKf/s/7B5v/L7FLv37P9pcY2e/WvbKKPZy/mf4f5B9bo/8+IffP/5M+W7X4J+O7Hz/ALG09v58TQTeTcRp5kbfeRsS8qcDIPBqxo3wN1u01FJtX0WfULVMH7PDfQ2+8gg4ZssdpGQQMHngjFfTlFHs5fzP8P8AIPrdH/nxD75//JnJf8JB4v8A+hH/APKtF/hXhF/8CvE9xqNzNZ6Z9ktZJXeG38+KTykJJVNxky2BgZPXFfUlFHs5fzP8P8g+t0f+fEPvn/8AJny5p/wQ8X2F9Hc/Yt+zPy+bEM5BHXzPevpfSdNh0bRrHS7dpGgsreO3jaQgsVRQoJwAM4HoKuUU4w5W23civivawjTjBRSbel93a+7fZBRRRWhyHjmr6RZ6tp2vaDr3h3xLNDNrk19b3GnWp+XnaGUtwcjcOQRhsjnBHk1j8P8AxxoOuR6jolnqcU1tKWtrg2cqPjkAsu1hyOq5YckcivryisY05RVlL8D0qmMoVZc86Wv+JnzP4zX4reOESDUNOuLazVNrWdla3EcUh3BtzggljkLjJwMcAEknrPhbp9v4C0dpLrwl4hm1243C5uo7EuoTd8qITghcBSeOW6kgLj2yinyT/m/Az9vhf+fP/kzPmv4kWHxA8caxKos9TOiRy+ZZ2j6fJF5eVA+YKG3MOfmLHqSAobaOPsfh1480y8jvLCx1O0uo87JoIp43XIIOGC5GQSPxr7Eoo5J/zfgHt8L/AM+f/JmeB/E+Txr43s7O003Tdas7B4lN7p8lkwQzKSciRV3SLz0YKPkVsZPHl/8Awqvxh/0Brv8A8BZv/iK+zKKOSf8AN+Ae3wv/AD5/8mZ518IbLxBY6HfJrkTpvmVw1wsgnklKjzXbf1U4TB6k789q9Fooq4R5Y2MMTX9vVdS1rhXhfxG8KeJfFuteKrXw9c5WOWya50/csf2oeV8p3kgfIRnaSAc56qoPulcl4f8A+SheMf8Aty/9FGoqfFH1/Rm+E/g1/wDAv/S4Hz7pnwB8c3/m/abex03Zjb9rug3mZznHlB+mO+OoxnnHd/Cv4Ra94X8WWfiKfXNNl0827kDT5nkF0rr8oJKqCnIfPPKrx3HulFanCfP/AI2+A/iTXvFuoavZa5Y3Ed7K0x+2tIjxZY7YxgPlVXaAcjpjAArAsP2f/Gtv4htk+3WNrDHsn/tK3nY+UwccIMK/mAfMOAvH3ga+n6KAPD/iZ8GvEnjLxpPrNhq1ibWWKNEhvJJFMG0YKrhWG0nLduXbjueQ/wCGcfGH/QS0P/v/ADf/ABqvp+igDi9L8PQ+HvhMvh3xPrUfkLZPaXV88wjWNZMqFV5OAFDhFJ9F4HSvjSvcP2kdZvH8Q6Toe/bYxWn2zYpI3yO7plhnBwE44yNzc816v8ItO07TfhnpUem3cF7HJ5kkl3DA0QmkLsG4YBjtxsDMASEHAGAAD5s034tePNKt2gt/Et26M5cm6VLhs4A4aRWIHHTOOvqauf8AC7fiH/0MP/klb/8Axuvqe+8J+G9TvJLy/wDD+lXd1JjfNPZRyO2AAMsRk4AA/CvH/jN8PfBPh3wXdaxZWH9n6nLdoLfyZJCkjsSWTYSVVdgdhgLjaAD2IB0lh4xufi98PNXtfDUsmieIIniVg1w6iEGQMGWVFyQyo44AOcgjGCfGJ/gz8Sbq4luLjRJJp5XLySSX8DM7E5JJMmSSec12n7Nejb9R1zXHSdfKiSzifGI33ne4zjlhsj6HgNyORX0PQB8//DHwb8VvDXiGyS4b7JokPE1rd3yywtGzguI40ZtsnJYHAGQcnBIPMeJ/hh8WPEeuTXur2X9pXC/uluFu4FRkXgbF3LtU8nG0dSSMk19T0UAfHlt8NPiRo/iC2istF1K11AoXiubWYKqAhgczq2xCQCMFgTkDuM/S/wARvCE3jjwdcaPb6hJZzl1ljIYiORlzhJAOShPPsQrYOMHrK5P4l6/feF/h5q+saY0a3kCRrE7ruCF5FTdjoSAxIzkZAyCOKAPkDX/Dmr+FtUbTdasZLS7CB9jEMGU9CrKSGHUZBPII6g19PfAr/knn/bz/AO0oq8F8I/DfxT8QNRW8EM8djcSl7jVbvJU5LbmGTmVsqw4z82NxXOa+jvhdpsOjaNq+l27SNBZatNbxtIQWKoqKCcADOB6Csp/HH5ndh/8Ada3/AG7+Z3NFFFanCFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABXO6n458OaNqMthf6j5N1FjenkSNjIBHIUjoRXRVyXh/8A5KF4x/7cv/RRrOpKSso9X+jOzCUqU1UnVTajG9k0vtRW7T79g/4WZ4Q/6C//AJLS/wDxFH/CzPCH/QX/APJaX/4iutqOeeG1t5bi4ljhgiQvJJIwVUUDJJJ4AA5zStV7r7v+CV7TA/8APuf/AIGv/lZy3/CzPCH/AEF//JaX/wCIo/4WZ4Q/6C//AJLS/wDxFfMPxU8Ww+M/Hl5qNnNJLp8SJb2ZeMIfLUcnHXBcuw3c4YZx0H0v8JtVvta+GGi3+pXMl1dukiPNJyzBJXRcnudqjk8nqcnmi1Xuvu/4Ie0wP/Puf/ga/wDlZP8A8LM8If8AQX/8lpf/AIij/hZnhD/oL/8AktL/APEV1tFFqvdfd/wQ9pgf+fc//A1/8rOS/wCFmeEP+gv/AOS0v/xFH/CzPCH/AEF//JaX/wCIrraKLVe6+7/gh7TA/wDPuf8A4Gv/AJWcl/wszwh/0F//ACWl/wDiKP8AhZnhD/oL/wDktL/8RXW0UWq9193/AAQ9pgf+fc//AANf/Kzkv+FmeEP+gv8A+S0v/wARR/wszwh/0F//ACWl/wDiK62sfxVoP/CT+F9R0X7bPZfbIjH58B+Zec8jupxhl4ypIyM5otV7r7v+CHtMD/z7n/4Gv/lZlf8ACzPCH/QX/wDJaX/4iprX4h+Fr28gtLfVN888ixxr9nlG5mOAMlcdTXyv43+GviDwF9mk1VIJrW44S6tGZ4w/PyElQQ2BnkcjOCcHD/hX/wAlD0n/AK+Yf/RqVM3Uir3X3f8ABNsPDBV6ipqElfrzp9P8CPsyiiitzywrLl8SaFBM8M2tadHLGxV0e6QMpHBBBPBrUrhPCOi6VqM3iSa+0yzupV1y6UPPArsB8pxkjpyfzrOcpJpR6nZhqNKcJ1Krdo22t1fmdL/wlPh7/oPaX/4GR/40f8JT4e/6D2l/+Bkf+NH/AAi3h7/oA6X/AOAcf+FH/CLeHv8AoA6X/wCAcf8AhS/e+RX+w/3/AMA/4Snw9/0HtL/8DI/8aP8AhKfD3/Qe0v8A8DI/8aP+EW8Pf9AHS/8AwDj/AMKP+EW8Pf8AQB0v/wAA4/8ACj975B/sP9/8A/4Snw9/0HtL/wDAyP8Axo/4Snw9/wBB7S//AAMj/wAaP+EW8Pf9AHS//AOP/CvE/jf4G1mG9i17w7YLFo9vZYu0sSIzGyuxLsgxkbWHIzgIc4ABo/e+Qf7D/f8AwPbP+Ep8Pf8AQe0v/wADI/8AGj/hKfD3/Qe0v/wMj/xr4f8A7Svv+f24/wC/rf419oaF4L0mx8PaZZ6jpGl3F9BaRRXE32ZX8yRUAZtxXJyQTk8mj975B/sP9/8AA0P+Ep8Pf9B7S/8AwMj/AMaP+Ep8Pf8AQe0v/wADI/8AGj/hFvD3/QB0v/wDj/wo/wCEW8Pf9AHS/wDwDj/wo/e+Qf7D/f8AwD/hKfD3/Qe0v/wMj/xo/wCEp8Pf9B7S/wDwMj/xo/4Rbw9/0AdL/wDAOP8Awo/4Rbw9/wBAHS//AADj/wAKP3vkH+w/3/wD/hKfD3/Qe0v/AMDI/wDGj/hKfD3/AEHtL/8AAyP/ABo/4Rbw9/0AdL/8A4/8KP8AhFvD3/QB0v8A8A4/8KP3vkH+w/3/AMA/4Snw9/0HtL/8DI/8aP8AhKfD3/Qe0v8A8DI/8aP+EW8Pf9AHS/8AwDj/AMKP+EW8Pf8AQB0v/wAA4/8ACj975B/sP9/8A/4Snw9/0HtL/wDAyP8Axo/4Snw9/wBB7S//AAMj/wAaP+EW8Pf9AHS//AOP/CuX+IegaNZeBdSuLTSLCCdPK2yRWyIy5lQHBAz0JFTOVSMXLTQ2w1HBV60KKclzNLp1djvqKKK3PLCiivNvB/h248QeFrLVLvxP4jSeffuWK/IUbXZRjIJ6Ad6znNpqKVzrw+GhUpyq1J8qTS2vq7v9D0miuS/4QT/qa/FH/gx/+xrH8VeFdU0rwvqN9ous+KNR1KGItBa/2j9855OAuWwMttHLYwOSKXPP+X8S/YYX/n9/5Kz0WivjP/hanjD/AKDN3/4FTf8AxdeqfCCXXvH1vq1xq+ua7BBavEkElrduquxDFwS24EgBDgdNwz1FHPP+X8Q9hhf+f3/krPeKK5L/AIQT/qa/FH/gx/8AsaP+EE/6mvxR/wCDH/7Gjnn/AC/iHsML/wA/v/JWdbRXJf8ACCf9TX4o/wDBj/8AY0f8IJ/1Nfij/wAGP/2NHPP+X8Q9hhf+f3/krOtorkv+EE/6mvxR/wCDH/7Gj/hBP+pr8Uf+DH/7Gjnn/L+Iewwv/P7/AMlZX1H4Z6Xqf29J9T1cWt9O1xPaLOvks5bdnYUwcEDGc9B6Vj/8KK8H/wDT3/5B/wDjddB/wgn/AFNfij/wY/8A2NH/AAgn/U1+KP8AwY//AGNZez/ufid31v8A6iP/ACU5/wD4UV4P/wCnv/yD/wDG62LH4bWemWcdnYeIPENpax52QwXgjRckk4ULgZJJ/GrH/CCf9TX4o/8ABj/9jR/wgn/U1+KP/Bj/APY0ez/ufiH1v/qI/wDJA/4QT/qa/FH/AIMf/saP+EE/6mvxR/4Mf/saPAL3H2PWre4vbq7+y6tPbxyXUpkfYoUAEn8/xNdbVwpwlFOxhicXiqFV0+e9vJf5HJf8IJ/1Nfij/wAGP/2NH/CCf9TX4o/8GP8A9jXW0VXsYdjD+0sV/N+C/wAjkv8AhBP+pr8Uf+DH/wCxo/4QT/qa/FH/AIMf/saztC1nx14g0a31S0Tw4kE+7asonDDaxU5wSOoPetH/AIuH/wBSv/5MVkvZtXUWejUeMpzcJ1oprR6rf7g/4QT/AKmvxR/4Mf8A7Gj/AIQT/qa/FH/gx/8Asa8s1b4+6zo2s32l3Gn2DT2VxJbyNHC5UsjFSRmQHGR6Cqf/AA0dqf8A0DbT/vw3/wAdotD+Vke0xX/P+H3r/I9f/wCEE/6mvxR/4Mf/ALGj/hBP+pr8Uf8Agx/+xo/4uH/1K/8A5MUf8XD/AOpX/wDJii0P5WHtMV/z/h96/wAg/wCEE/6mvxR/4Mf/ALGj/hBP+pr8Uf8Agx/+xo/4uH/1K/8A5MUf8XD/AOpX/wDJii0P5WHtMV/z/h96/wAg/wCEE/6mvxR/4Mf/ALGj/hBP+pr8Uf8Agx/+xo/4uH/1K/8A5MUf8XD/AOpX/wDJii0P5WHtMV/z/h96/wAg/wCEE/6mvxR/4Mf/ALGj/hBP+pr8Uf8Agx/+xo/4uH/1K/8A5MUf8XD/AOpX/wDJii0P5WHtMV/z/h96/wAg/wCEE/6mvxR/4Mf/ALGj/hBP+pr8Uf8Agx/+xo/4uH/1K/8A5MUf8XD/AOpX/wDJii0P5WHtMV/z/h96/wAg/wCEE/6mvxR/4Mf/ALGj/hBP+pr8Uf8Agx/+xo/4uH/1K/8A5MUf8XD/AOpX/wDJii0P5WHtMV/z/h96/wAg/wCEE/6mvxR/4Mf/ALGj/hBP+pr8Uf8Agx/+xo/4uH/1K/8A5MUf8XD/AOpX/wDJii0P5WHtMV/z/h96/wAg/wCEE/6mvxR/4Mf/ALGj/hBP+pr8Uf8Agx/+xo/4uH/1K/8A5MUf8XD/AOpX/wDJii0P5WHtMV/z/h96/wAjX8P+H7fw5Zz29vcXVx587XEkl04d2dgASSAM9M/nWtXJf8XD/wCpX/8AJij/AIuH/wBSv/5MVpGairKLOWrhp1Zuc6sW35/8A62iuS/4uH/1K/8A5MVN4Y1nWb3WdZ0vWksBPp/kYayD7W8xS38Rz0A7DvVKqrpNPUylgZKnKpGcXy6uz1tdL82jp6KKK0OIKKyfE2t/8I54eutV+z/aPI2fut+zducL1wcdc9KyP+Eg8X/9CP8A+VaL/Cs5VIxdn+TZ10sFVqw9pGyV7ayjHVWvu13R1tFePar8ebbRdUuNNv8ARY0u7Z9kqJfeYFbuNyRkZHQjPByDyDWp4b+K914u8waH4egu5I8lof7VjjkAGMtsdQ235gN2MZOM5pe2j5/c/wDIv+zq3eH/AIMh/wDJHptFcl/wkHi//oR//KtF/hR/wkHi/wD6Ef8A8q0X+FHto+f3P/IP7Ord4f8AgyH/AMkdbRXJf8JB4v8A+hH/APKtF/hR/wAJB4v/AOhH/wDKtF/hR7aPn9z/AMg/s6t3h/4Mh/8AJHW0VyX/AAkHi/8A6Ef/AMq0X+FH/CQeL/8AoR//ACrRf4Ue2j5/c/8AIP7Ord4f+DIf/JHW0VyX/CQeL/8AoR//ACrRf4Uf8JB4v/6Ef/yrRf4Ue2j5/c/8g/s6t3h/4Mh/8kdbRXJf8JB4v/6Ef/yrRf4Uf8JB4v8A+hH/APKtF/hR7aPn9z/yD+zq3eH/AIMh/wDJHW0VyX/CQeL/APoR/wDyrRf4Uf8ACQeL/wDoR/8AyrRf4Ue2j5/c/wDIP7Ord4f+DIf/ACR1tFcbJ4v16yvNPj1Twp9jgvLuO0Wb+0Uk2s5/uquegJ7dK7KqjNS2Ma+FqULOdtdrNNfemwoooqznCiuNtfiLb3tulxaeHPEc8D52yRWIdWwcHBDY6gipf+E7/wCpU8Uf+C7/AOyrL29Pud7yzFp2cPxX+Z1tFcl/wnf/AFKnij/wXf8A2Vc//wAL18H/APT3/wCQf/jlHtodxf2biv5fxX+Z6bRXBaZ8WNH1vzf7J0rWr/yceZ9kt0l2ZzjO1zjOD19DWh/wnf8A1Knij/wXf/ZUe2h3D+zcV/L+K/zOtorkv+E7/wCpU8Uf+C7/AOyo/wCE7/6lTxR/4Lv/ALKj20O4f2biv5fxX+Z1tFcl/wAJ3/1Knij/AMF3/wBlR/wnf/UqeKP/AAXf/ZUe2h3D+zcV/L+K/wAzraK4+X4gRwQvNN4Y8SxxRqWd3sMKoHJJJbgV1Gn3keo6ba30KssVzCkyBxhgGAIzjvzVRqRk7JmVbCVqMVKpGyZYoooqzmCiuYuviH4Wsrye0uNU2TwSNHIv2eU7WU4IyFx1FQ/8LM8If9Bf/wAlpf8A4is/bU/5l952rLca1dUZf+Av/I62iuS/4WZ4Q/6C/wD5LS//ABFH/CzPCH/QX/8AJaX/AOIpe3pfzL7x/wBl47/nzP8A8Bf+R1tFcl/wszwh/wBBf/yWl/8AiKP+FmeEP+gv/wCS0v8A8RR7el/MvvD+y8d/z5n/AOAv/I62iuS/4WZ4Q/6C/wD5LS//ABFH/CzPCH/QX/8AJaX/AOIo9vS/mX3h/ZeO/wCfM/8AwF/5HW0VyX/CzPCH/QX/APJaX/4ij/hZnhD/AKC//ktL/wDEUe3pfzL7w/svHf8APmf/AIC/8jraK5L/AIWZ4Q/6C/8A5LS//EUf8LM8If8AQX/8lpf/AIij29L+ZfeH9l47/nzP/wABf+R1tFcl/wALM8If9Bf/AMlpf/iKP+FmeEP+gv8A+S0v/wARR7el/MvvD+y8d/z5n/4C/wDI62iuS/4WZ4Q/6C//AJLS/wDxFH/CzPCH/QX/APJaX/4ij29L+ZfeH9l47/nzP/wF/wCR1tFcl/wszwh/0F//ACWl/wDiKP8AhZnhD/oL/wDktL/8RR7el/MvvD+y8d/z5n/4C/8AI62iuS/4WZ4Q/wCgv/5LS/8AxFH/AAszwh/0F/8AyWl/+Io9vS/mX3h/ZeO/58z/APAX/kdbRXJf8LM8If8AQX/8lpf/AIij/hZnhD/oL/8AktL/APEUe3pfzL7w/svHf8+Z/wDgL/yOtorkv+FmeEP+gv8A+S0v/wARR/wszwh/0F//ACWl/wDiKPb0v5l94f2Xjv8AnzP/AMBf+R1tcPpuq6dpnxC8W/b7+1tPM+x7PPmWPdiI5xk84yPzq3/wszwh/wBBf/yWl/8AiKzrrxb8N724e4u/sE8743SS6a7s2BgZJjz0AFZ1KsHZxktPPyZ2YTBYqmqkatCdpK2kXf4ovqvI6j/hKfD3/Qe0v/wMj/xo/wCEp8Pf9B7S/wDwMj/xrkv+Ej+F3/PDS/8AwVN/8bo/4SP4Xf8APDS//BU3/wAbpe3/AL0fvK/sv/pzV/8AAf8AgHW/8JT4e/6D2l/+Bkf+NH/CU+Hv+g9pf/gZH/jXJf8ACR/C7/nhpf8A4Km/+N0f8JH8Lv8Anhpf/gqb/wCN0e3/AL0fvD+y/wDpzV/8B/4B1v8AwlPh7/oPaX/4GR/40f8ACU+Hv+g9pf8A4GR/41yX/CR/C7/nhpf/AIKm/wDjdH/CR/C7/nhpf/gqb/43R7f+9H7w/sv/AKc1f/Af+AYnxe0Hw9450NLnTdZ0Qa7Z8wu93Gpnj5zCW3YGScgtkA8fKGY14noHjrx94X0tdM0e9vILNXLrE1okoQnrt3oSBnnA4ySepNfQ/wDwkfwu/wCeGl/+Cpv/AI3R/wAJH8Lv+eGl/wDgqb/43R7f+9H7w/sv/pzV/wDAf+AeWf8AC/fH3/QA0v8A8A5//jlcfqd54w+KHiGL+1rmGLZny/tbra21qjOM43YzjI6bnIUfe219SaZpPhLWdOiv7DR9LmtZc7H+woucEg8FQeoNXP8AhFvD3/QB0v8A8A4/8K1TqNXVjjlHBwk4yU016GH4Pk8MeD/CdhoMHibT7hLRGBme6iBdmYuxwDwNzHA5wMcnrW5/wlPh7/oPaX/4GR/40f8ACLeHv+gDpf8A4Bx/4Vy9/e+FbLWbvS4/BMt9PabPNay0qKVV3KGHv0PcdjUynOOsrfiaUMPhq7caam7a/Z9P1Oo/4Snw9/0HtL/8DI/8aP8AhKfD3/Qe0v8A8DI/8a5L+0fD3/RNtU/8Ecf+NH9o+Hv+ibap/wCCOP8AxqPbPuvxOj+zaf8ALP74/wCZ1v8AwlPh7/oPaX/4GR/40f8ACU+Hv+g9pf8A4GR/41yX9o+Hv+ibap/4I4/8aP7R8Pf9E21T/wAEcf8AjR7Z91+If2bT/ln98f8AM63/AISnw9/0HtL/APAyP/GsX4fyxzw+IpoZFkik1y5ZHQ5VgdpBBHUViWPiDwlqdnHeWHgG9u7WTOyaDR4pEbBIOGBwcEEfhWtZ+K7LToTDY+CfEFrEzbikGlBFJ6ZwD14H5UKpeScmtPUqWDcKM6dKErytu49PRna0VyX/AAnf/UqeKP8AwXf/AGVH/Cd/9Sp4o/8ABd/9lWvtodzz/wCzcV/L+K/zOtorkv8AhO/+pU8Uf+C7/wCyo/4Tv/qVPFH/AILv/sqPbQ7h/ZuK/l/Ff5nW0VyX/Cd/9Sp4o/8ABd/9lR/wnf8A1Knij/wXf/ZUe2h3D+zcV/L+K/zOtorkv+E7/wCpU8Uf+C7/AOyo/wCE7/6lTxR/4Lv/ALKj20O4f2biv5fxX+Z1tFcl/wAJ3/1Knij/AMF3/wBlR/wnf/UqeKP/AAXf/ZUe2h3D+zcV/L+K/wAzraK5L/hO/wDqVPFH/gu/+yrn/wDhevg//p7/APIP/wAco9tDuH9m4r+X8V/mem0Vk+H/ABBb+I7Oe4t7e6t/Ina3kjukCOrqASCATjrj861q0jJSV0ctWlOlNwmrNBRRRTMwooooAKKKKACiiigArkvD/wDyULxj/wBuX/oo11tcl4f/AOSheMf+3L/0UayqfFH1/RndhP4Nf/Av/S4HW1x/xT1m80D4Z63qNg/l3SxLEkgJBTzHWMspBBDAOSD2IFdhXy5+0Lr7al48i0dWk8jSrdVKMqgebIA7MpHJBXyhz3U4Hc6nCeR19H/s16n5vh7XNJ8nH2a7S583d97zU27cY4x5Oc553dsc4HxD+DuneFvhfaataHGrWHl/2lKZ2ZJ95CnaCP4XZQvC/LnOTiuL+Dmvt4f+JulNuk8i+f7DMqKrFhIQEHPQCTyySOcA9ehAPsOiivnD416t8QNK8UG+E99pWiHFtZyaffSCOXALbn2kYkO48EDhcDcF3EA+j6K+OPD3xc8a+HrwzprM+oRv9+DUna4RsAgck7l65+UjOBnIGK6CGy+N3ie8ubpH8RwyDbvDzmwTpgbUJRT93naPc8nkA+p6K+ePhJ8VdUs/EP8Awivi+6vp2nlW2tZLkbpLefewMcmRvO5mC5YnaVAwBkj6HoAKK+f/AIkfHm4t9Rl0nwZLAY4vll1MoJNzgjiIH5SowQWIO7J24ADHE02H466rftrdudZSVXKGO6kjt487AOLeQqhGD12Yzk9QaAO7/aO/5J5p/wD2FY//AEVLXk/wk02GXWYdUZpPPt9WsLdFBG0rI0jMTxnOYlxz3PXtU+I3jXxfrr23h/xVBHa3GlORLHGjIZZCoG9xuKMcZKlQBhzjg1P8JdM83xDbat52Ps2p2Vt5W373muzbs54x5OMY53dsc5VvgZ3Zb/vUfn+TPryiiitThCuS8Cf8zL/2Hrr/ANlrra+W/GHxE8VeEfGWtWGh6r9ktZL6eZk+zxSZcyMCcupPRR+VZT+OPzO7D/7rW/7d/M+pKK+cPht8V/iFr/i200h/I1aGeWM3DSWyobaAN+8cFNgHB/izztAGTg+n/Fjxp4g8EaHa6jomkwXcLSlbq4nDMluOAoKqQfmJ+9nAxjqwrU4T0CivmD/ho7xh/wBA3Q/+/E3/AMdr6H8K6hqmq+F9Ovta07+ztSmiDT2ufuHPBweVyMNtPK5weQaANiivmD/ho7xh/wBA3Q/+/E3/AMdr6T0m8m1HRrG9uLSSznuLeOWS2kzuhZlBKHIByCcdB06CgD4Mr7/r4Y8J31vpnjLQ7+8k8u1tdQt5pn2k7UWRSxwOTgA9K931n9oB7zUX0vwT4en1S6bIhmmVjvKklisKfMy7BkEspGeRxyAe4UV4X4G/aDXU9Uj07xXaWliJ3Kx31uzLEh42q6sSQM5+fdgZGQBlh7pQAUVj+J/E+l+EdDm1fV5/Kt4+FVeXlc9EQd2OD+RJIAJHkepftLabFcKul+G7u5g2As91crAwbJ4CqHBGMc57njjkA90oryPwZ8e9F8R6o9hrFpHoRKboZ57sPE5GSVZyqhDjpng8jIOAe48a+NdN8B6NDqmqQXc0Etwtuq2qKzBirNk7mUYwh7+lAHSUV4vN+0j4bW8tlh0bVXtW3faJHEavHx8u1QxD5PXLLjrz0o8WftDaNp22Hwza/wBryPExNxLvhjhfoo2soZ/Uj5eMYOScAHtFcl8TP+Se6p/2y/8ARqV5x4X/AGjbGdEg8UaZJazl1X7TYjfFgscsyMdyhRt6FyeeBwK9D+Is8N18Nb+4t5Y5oJUgeOSNgyuplQggjggjnNZV/wCFL0Z3ZX/v1H/HH80djRRRWpwhXnXh3Xv+EY+CsetfYp737HFNJ5EA+Zv3zDk9lGcs3OFBODjFei18x+KfiJpcfwlXwRBDPNqUuDPJjbHCPPMoGTyzYC8AYw/XIIrJ/wAVej/Q7qf+41P8cPyqHUfDf42a94s8b2ehapp2mrBdpIFktVdGRlQvk7mYEYUjHHUHPGD7pXxp8K/EukeE/Hlnqus28kluqPEsyE5tmcbfM2j74ClgR6MSASAD9P6B8S/B/ijVF0zR9ajnvGQusTQyRFwOu3eoBOOcDnAJ6A1qcJ4R+0VBDD8RrV4oo0ebTInlZVALt5ki5b1O1VGT2AHavQ/2cf8Aknmof9hWT/0VFXIftKf2j/wkOh+b/wAgz7I/2f7v+u3/AL3/AGvu+T149O9df8LfHPhbw58IdJXVtesbeaDzvMt/NDzLuuHx+6XLngg8Dpz0oA9gorx//ho7wf8A9A3XP+/EP/x2vWLG/s9Ts47ywu4Lu1kzsmgkEiNgkHDDg4II/CgCxRRXD+J/i34O8KyzW11qf2q+h4a0sl81wd20qTwisCDlWYEY6cjIB3FFeH2v7SmjvqM6Xnh++isRu8maGZJJH5+Xch2hcjJOGbB45617Bo2uaX4h05L/AEi/gvbVsDfC+dpIB2sOqtgjKnBGeRQBoUVy/iH4i+EfCt4LPWNbgguj1hRXldOAfmVASuQwI3Yz2zR4b+InhXxdqMlhoeq/a7qOIzMn2eWPCAgE5dQOrD86AOoorP1nXNL8Pac9/q9/BZWq5G+Z8biATtUdWbAOFGSccCuP/wCF2/Dz/oYf/JK4/wDjdAGn4E/5mX/sPXX/ALLXW1yXgT/mZf8AsPXX/stdbWVH4Ed2Zf71L5fkgooorU4Tkvhn/wAk90v/ALa/+jXrra5L4Z/8k90v/tr/AOjXqDUvi14D0q4WC48S2juyBwbVXuFxkjlo1YA8dM56eorKh/Cj6I7s0/36t/jl+bNjVfBvhnXHuJdT0DTbme4TZLO9svmsNu3/AFmNwIGACDkYGMYr4s0LTP7b8Q6ZpPneT9uu4rbzdu7ZvcLuxkZxnOMivs/w3448NeLvMGh6vBdyR5LQ4aOQAYy2xwG2/MBuxjJxnNfIHgT/AJKH4a/7Ctr/AOjVrU4T7foorP1PXdH0Tyv7W1WxsPOz5f2u4SLfjGcbiM4yOnqKANCiuf8A+E78H/8AQ16H/wCDGH/4qugoAKKKrw39ncXlzZw3cEl1a7ftEKSAvFuGV3KOVyORnrQBYqOeeG1t5bi4ljhgiQvJJIwVUUDJJJ4AA5zRJPDC8KSyxo8z7IlZgC7bS2F9TtVjgdgT2rD8d/8AJPPEv/YKuv8A0U1AGXpXxZ8D61qlvpthr0b3dy+yJHgljDN2G51AyegGeTgDkiu0r4c8G6baax410TTr9oxaXN7FHKHLgOpYZTKDILfdB4wSMkDJH3HQAUUVzev+P/Cnhd2i1jXLSCdXCNApMsqEruG6NAWAxzkjHI9RQB0lFY/h7xVoXiuzN1oepwXsa/fCEh48kgbkOGXO04yBnGRxWxQAUVT1LVtN0a3W41TULSxgZwiyXUyxKWwTgFiBnAJx7Gq9n4l0HULe6uLLW9NuYLRN9zJDdI6wrgnLkHCjCk5PofSgDUrm/Evj7wv4QuILfXdWjtZ50LxxiN5G2g4yQikgZyATjODjoa3LG/s9Ts47ywu4Lu1kzsmgkEiNgkHDDg4II/Cvkj42/wDJXtd/7d//AEnjoA+q9A8R6R4p0tdS0W+ju7QuU3qCpVh1DKwBU9DggcEHoRWN4f8A+SheMf8Aty/9FGuM/Zx/5J5qH/YVk/8ARUVdn4f/AOSheMf+3L/0UayqfFH1/RndhP4Nf/Av/S4HW0UUVqcJyXxM/wCSe6p/2y/9GpVD4veLbvwd4DlvdOmkg1Ce4jt7aZY0cIxO9iwbIxsRx0PJH1F/4mf8k91T/tl/6NSvNP2l764j07w7YLJi1mlnmkTaPmdAgU568CR/z9hWS/iv0X6ndU/3Gn/jn+VM4T4QfDbTfiDcas2qXl3BBYpEFS12qzs5bkswIAAQ8Y5yORjns/hp8K/F3hL4oC+nTytGtvPjNz56L9rjIKp+7VmPJ2PtbgbeuQMn7Mv/ADNP/bp/7Wr6ArU4QooooAKKKKACiiigAooooAKKKKACiiigDkvHf/Mtf9h61/8AZq62uS8d/wDMtf8AYetf/Zq62sofHL5HdX/3Wj/29+YUUUVqcJyXwz/5J7pf/bX/ANGvXQx6tps2qTaXFqFo+oQpvltFmUyovHLJnIHzLyR3HrXBaNPNa/AO9uLeWSGeLTL545I2KsjAykEEcgg85r5RsPtn9o239nef9u81Ps/2fPmeZkbdmOd2cYxzmsqH8KPojuzT/fq3+OX5s+96+NPiN8Ob74eapbQT3Ud5Z3aFra5VdhcrjerJklSCw7kEEc5yB9l143+0hBC3gXTLhoozOmpqiSFRuVWikLAHqASqkjvtHpWpwnCfs4/8lD1D/sFSf+jYq+n6+dP2abyZNZ1+yW0kaCa3ile5GdsbIzBUPGMsJGI5H3Dwe30XQAUUUUAFFFFAGT4p/wCRQ1r/AK8J/wD0W1Hhb/kUNF/68IP/AEWtHin/AJFDWv8Arwn/APRbUeFv+RQ0X/rwg/8ARa1l/wAvfkd3/MD/ANv/AKGtRRRWpwnJeBP+Zl/7D11/7LXW1yXgT/mZf+w9df8AstdbWVH4Ed2Zf71L5fkgooorU4QooooAK8X8cfHn/hGPEeqaFYaLBeyWn7tLw3uU8woCdyBf4WJUruB+UjIPT2ivnD4l/BvxdqXjS/1fR4/7Utb+UyjzLtBLDwvytv2jaDkKFJwqgHHcA9v8D+JP+Eu8F6Xrhj8uS6i/eoFwBIpKPtGT8u5Wxk5xjPNdBXF/C7wVN4D8HDS7ueOa8luHuLhonLRhjhQEyqnG1Ezkdc9sV2lABRRRQAUUUUAFeD/Fr4R+KfFHii68QaTdQX0LRRJHZSzFJI8AKVTd8m3OX5ZeWbjPX3iigD4Ar7f8Cf8AJPPDX/YKtf8A0UtfHHiyxt9M8Za5YWcfl2trqFxDCm4naiyMFGTycADrX2n4b0SHw54a03RoPLKWdukRdIxGJGA+Z9o6Fmyx5PJPJoA1KKKKACiiigAooooAKKKKACiiigAooooAKKKx/FXiG38KeF9R1y6XfHaRFwmSPMcnCJkA43MVGccZyeKANiivjSf4seN5vEEusJ4gu4JXcstvG5NugIxtETZTAHqCc8kk819l0Acl8M/+Se6X/wBtf/Rr11tcl8M/+Se6X/21/wDRr11tZUP4UfRHdmn+/Vv8cvzYVyXh/wD5KF4x/wC3L/0Ua62uS8P/APJQvGP/AG5f+ijRU+KPr+jDCfwa/wDgX/pcDraKKK1OEK+QPiH4D8baXqOp6/r8E97atdlDqZeM+YCdqOUViY1ICgAgBcqvoK+v68/+Nv8AySHXf+3f/wBKI6APlzwWJm8deHlt5I45zqdsI3kQuqt5q4JUEEjPbIz6ivuOviDwJ/yUPw1/2FbX/wBGrX2/QAUUUUAFFFFABRRRQAUUUUAFc/44tddvvBeqW3hq48jV5IsW7hwh6jcFY/dYruAPGCQcjqOgooA+CL/7Z/aNz/aPn/bvNf7R9oz5nmZO7fnndnOc85qvXtH7RdjcSeMra/WPNrDp9vDI+4fK7yXBUY68iN/y9xXi9AH2N8LtSh1nRtX1S3WRYL3VpriNZAAwV1RgDgkZwfU13Ncl4E/5mX/sPXX/ALLXW1lR+BHdmX+9S+X5IKKKK1OEKKKKACiiigAooooAK5Lw/wD8lC8Y/wDbl/6KNdbXJeH/APkoXjH/ALcv/RRrKp8UfX9Gd2E/g1/8C/8AS4F3xx4k/wCER8F6prgj8yS1i/dIVyDIxCJuGR8u5lzg5xnHNfHGjDxBqHiFL/SIL6/1eCUXu+GFriQOrg+Ywwc/MRknOSeetewftE+MLhtRtPCdnc7bVIhcXqxSg+Y5PyI4AyNoUPgnnepxwDWv+zn4YuLHR9R8R3UEAj1DbFZycGTYjOJP91S20YzyUyRgKTqcJ5hq/jT4leKdH1bSdRe+vLG2x/aMS6ai+Rsbd+8KxgpgoTyR90+9ef19/wBfGHxU0z+yPih4htvO83fdm53bduPOAl24yem/Ge+M8dKAPr/QtT/tvw9pmreT5P260iufK3btm9A23OBnGcZwK4/42/8AJIdd/wC3f/0ojrm/2ddfbUPB19osrSM+l3AaPKqFWKXLBQRyTvWUnP8AeHPYdJ8bf+SQ67/27/8ApRHQB4R8CrG4u/ivps0Ee+O0inmnO4DYhjaMHnr8zqOPX0zX1vXzB+zj/wAlD1D/ALBUn/o2Kvp+gD4g8d/8lD8S/wDYVuv/AEa1fV/xT1m80D4Z63qNg/l3SxLEkgJBTzHWMspBBDAOSD2IFfGkEE11cRW9vFJNPK4SOONSzOxOAABySTxivrv406J/bfwv1TZb+dcWO29i+fbs2H526gHEZk4OfYZxQB8+fBzQG8QfE3Sl2yeRYv8AbpmRlUqIyCh56gyeWCBzgnp1H2HXyp+z7cwwfE0Ry3UkL3FlLHFGqgidgVbYxwcDarNkFeUAzzg/VdAHh/7SOjWb+HtJ1zZtvorv7HvUAb43R3wxxk4Kcc4G5uOa4P4P/wAf/Ye0r/24rrP2mv8AmVv+3v8A9o1l/Cixt4/BthfrHi6m8XWsMj7j8yJHlRjpwZH/AD9hWVb4Gd2W/wC9R+f5M+lKKKK1OEK8i1Pwj/wmngbxlpsS7r6LXrm5sucfvkAwv3gPmBZMk4G7PavXa5LwJ/zMv/Yeuv8A2Wsp/HH5ndh/91rf9u/mfKvw88Tv4R8c6Zqnn+TaiURXhO4qYGOHyq8tgfMBz8yqcHFfZeralDo2jX2qXCyNBZW8lxIsYBYqiliBkgZwPUV8kfGPQG8P/E3VV2yeRfP9uhZ2ViwkJLnjoBJ5gAPOAOvU7Hiv4ozeM/h9oPhO0tNSbVA8SXkhlMn2pkG1QAPmkLsQ5BAwwH3utanCeV1972F9b6np1tf2cnmWt1Ek0L7SNyMAVODyMgjrXz58evAekeH9G0PVNE060sYI3NjOsWQ0ny7oyezEBJMsTuORknth6X8WLfS/glN4Shs86mfNtN0hJRoJjIzyDA+8u7btJ/iVsnlQAaHw/t7z4o/GifxfdWMEVjZSpczRiUjYyoVt1HdmzGrE8KdjZxkKfpeuP+GPhH/hC/A1lpsq7b6X/Sb3nP75wMr94j5QFTIODtz3rsKAPgzSdNm1nWbHS7do1nvbiO3jaQkKGdgoJwCcZPoa+39A8OaR4W0tdN0WxjtLQOX2KSxZj1LMxJY9Bkk8ADoBXxp4E/5KH4a/7Ctr/wCjVr7foA+XP2ioIYfiNavFFGjzaZE8rKoBdvMkXLep2qoyewA7V7P8NNfVvg1pGsam0cMFlZSLK6KxCxQFk3Y5JO2ME46nOB2rxz9o7/koen/9gqP/ANGy16npkl9L+ze7ajDHDOPDk6qqHIMQhYRN1PJjCE+5PA6AA+ePiD45vvHXiWa+nlkFhE7JYW5G0QxZ4yASN5ABY5OTx0AA+k/Cc/wz8Fac1noviDQ4vN2meZ9TieSZlGAWYt9TgYUEnAGTXzB4H0Gz8T+NNL0a/vfsVrdy7HmBAPAJCrnjcxAUdeWHB6H3/wD4Zx8H/wDQS1z/AL/w/wDxqgDzT40aZ4IsbjSp/B0mms9y9w94LC7Eqg5Qr8oYhBy+AAB+XHq/w2g0j4h/BrTdN16KPUks3NtKjKYzE0Z/dhWXBBETINynkEgk5aqf/DOPg/8A6CWuf9/4f/jVegeDvDOheFND/s7w+M2vmsZZDMZTJMuI3LHOA2UwQMAEHgUAfGniXTYdG8Vavpdu0jQWV7NbxtIQWKo5UE4AGcD0FfUfgv4N+GvD2j2v9raXY6nq/lFbm4mRpY2JbdhUclRjhQwUEgZ4yRXzR47/AOSh+Jf+wrdf+jWr7foA+UPjxoel6D4+t4tJsILKGfT45pIoE2Jv3yLkKOF4RegHr1JNdX4Fvri7/Zz1mGeTfHaah5MA2gbELwyEcdfmdjz6+mKxP2jv+Sh6f/2Co/8A0bLWn8Pf+TevEn/YVH/tvWVf+FL0Z3ZX/v1H/HH80fRlFFFanCFfKfjrwdpdp8M9G8XQeempXd39lnXfmNwDNhsEZDYRRwcYHTOTX1ZXzH8SNT8r4OeD9J8nP2m7ubnzd33fKd124xznzs5zxt754yf8Vej/AEO6n/uNT/HD8qhifBTwto3i7xleWGuWf2u1j095lTzXjw4kjAOUIPRj+dfT/h7wroXhSzNroemQWUbffKAl5MEkbnOWbG44yTjOBxXi/wCzL/zNP/bp/wC1q+gK1OE+f/2mv+ZW/wC3v/2jWZ8E/hjoHi3Rr/Wtejku0S4NpFah2jVSFRy5ZSCT8wAHAGD1yMSftKan5viHQ9J8nH2a0e583d97zX27cY4x5Oc553dsc+n/AAS/5JDoX/bx/wClElAGH8TfhR4Xbwdrmr6To1pZapBb/aFkjd441WPDOBGp2AlFYfd5JyfWuE/Z28TvY+KLvw5PPi11CIywRtuP79Bk7ccLlNxJI52KM8AH3P4gzw23w58SPPLHEh0y4QM7BQWaMqo57liAB3JArwT9nH/koeof9gqT/wBGxUAel/HPxtD4e8HTaLbXEf8Aamqp5XlggtHbnIkcggjBAKDOD8xIPymuA+Cfwx0Dxbo1/rWvRyXaJcG0itQ7RqpCo5cspBJ+YADgDB65GKf7R3/JQ9P/AOwVH/6Nlr2f4P2Nxp/wo0CG6j8uRonmA3A5SSR5EPHqrKfbPPNAEmt/CfwRrlv5U3h+0tXVHWOWwQW7IWH3vkwGIwCNwYD05Ofnz4VazqPgr4q2+k3Tzwx3F2dOvrWMq4aTJRM84+WQg7gc43YyCQfrevkDW7fTta+Pk9k1j5Vjc+IFtp4PNZvMzMEkbdwRvO5sD7u7A6CgD0/46/DvS5NH1LxvBNPDqUXkCePO6OYbliBweVbBXkHGE6ZJNecfAr7H/wALX037T5/neVP9l8rG3zPLbO/P8Ozf053be2a93+Nv/JIdd/7d/wD0ojrxz9nWCGb4jXTyxRu8OmSvEzKCUbzI1yvodrMMjsSO9AHu/jj4eaL4/t7OLVnu4ns3ZoZbWQKwDAblO4EEHap6Z+UYI5z8ea7pn9ieIdT0nzvO+w3ctt5u3bv2OV3YycZxnGTX3fXwx4svrfU/GWuX9nJ5lrdahcTQvtI3I0jFTg8jII60AfVXwn0z+xPD2paT53nfYdTltvN27d+xEXdjJxnGcZNd7XHfD2Fba38QwIZCket3KKZJGdiAEHLMSWPuSSe9djWVH4Ed2Zf71L5fkgooorU4T5T8WfEi8t/BOn+DNJm8mMxSHUZkyHbdI5EI44UqVJIJ3btvADA6Pwu+Ctt4x8PjXtbvru3tJndLaG12BnCkDzC53YG4Ou0qDwDnHXy7xB/yHLn/AID/AOgivszwJ/yTzw1/2CrX/wBFLWVD+FH0R3Zp/v1b/HL82eT+MP2drNrO4vPCd5Ol0u+RbC6YMknIIRH4KYGQN27PGSOTXkHw4sbjUPiV4chtY/MkXUIZiNwGEjYSOefRVY++OOa+16+TPhRHYw/Hewi0uaSfT0uLtbWWQYZ4hDLsY8Dkrg9B9BWpwn0v4w8Sw+D/AAnf69PbyXCWiKRChALszBFGT0G5hk84GeD0r5o8BeCr74v+INb1LWNZkR4kDzzld8jyyBhHgcAIuzkDHChRjOV+p9V0qx1zS7jTNTto7mzuE2SxP0YfzBBwQRyCARgivG9G8VfC74RRX1ro+p32sXVzLH9oMBWc7QpK4cbIio3H7pLZbB6cAHIfEf4I/wDCH+HBrmm6p9ptbWJBepcja7SM6pujCjG0lvuscrj7zZ49L+A/i2bxF4KfTryaN7vSHS3ULGVIt9o8osehPyuvHOEGeTk8Z8WviVrGreF7rR/+EM1XStMu5Yk+3anC8TPtIk2hcbVbcn95vlU8c8T/ALMv/M0/9un/ALWoA6T4y/DW+8Xo2u22tRwjTbI7LG4+SIkMWd/MLBUJXAyRj5FyQOR8yWN/eaZeR3lhdz2l1HnZNBIY3XIIOGHIyCR+Nfe9fEHgT/kofhr/ALCtr/6NWgDY8Q/Dr4ix3gvNY0TVb66uesyN9sdtoA+ZkLEcYA3enHSvV/id4X8c3/w101xqubSw0qGTWNPlkHmSTRKC8nmc+Z3JUtjMYI3MRj3Cuf8AHf8AyTzxL/2Crr/0U1AHxx4Y8Map4u1yHSNIg824k5Zm4SJB1dz2UZH5gAEkA+5+GfA/xX0jx5pL6n4ku7rSUfzbqddQa4i2gNmNo5SpJbAGQp27wwORxwHwGmvovipZpaCQwS28yXm2PcBFsLDccfKPMWPnjnA74P1nQB4n8Z/izNoLzeFdAeSLUyg+2XgBU26soIWM/wB8qQdw+6Dx83K8h4L/AGf9X1q3W98R3EmjwF0ZLYRh55YyAWzziI4OBuBIOcqMc+V67qf9t+IdT1byfJ+3Xctz5W7ds3uW25wM4zjOBXu/h6T48aBZm1fSINUj/gOpXUMrpySfnEys2c/xE4wAMCgDzzxx8PNe+FV/Z6laarJJBK7Jb39oHhkjbYMhscISGcABjkK3TkV9B/Czx0njrwlHcSnGp2e2C9UsuWcKP3oC4wr8kcDBDAZ25PknjPwx8ZvHTour6PGtpE++Kzt7q3SJG2gZx5hLHryxONzYwDiuz+BfgnxF4O/t7+39P+x/avs/k/vo5N23zN33GOMbl6+tAHkmv3a/Fb4ytDBqMdpZ3lwLW0nuXYokSDAKggEFyCwTj5pMZySa6TxV+z9ceHvC+o6xa6/9vksojMbf7GItyKcudxkOMLuboc4wOTXL/FP4b3ngbXJLiGHfoV3Kxs5kyRFnJELZJIYDoSfmAz13AdB4A+O+qaBssPEvn6rpo3kXGd92hPIG5mAdc54PI3dcKFoA2P2fbPVJPD3jOTTT9nupooorK6lT92s4SXHOCDtLISMHgjjmvG/EegX3hbxBeaLqSxi7tXCv5bblYEBlYH0KkHnB55APFfZfg698Lahof2rwilimmvK24WcAhAkGAdyYBDYC9RkjaemK+WPjBY2+n/FfX4bWPy42lSYjcTl5I0kc8+rMx9s8cUAR+GvAXjzxB4fnn0Gwu30m8cJKBdJDHcGM5GVZ13hWzg4IBB7g17N8EfD2s+HrzVbDU2+yyWsSJcWWEfc7szxvvUnGF3cDr5nPK10vwS/5JDoX/bx/6USVp+H/APkoXjH/ALcv/RRrKp8UfX9Gd2E/g1/8C/8AS4HW0UUVqcJyXxM/5J7qn/bL/wBGpXzT8TPE3jLxLcadL4s0eTS0hSRbWI2UkCsSV3sPMyWP3AecDA4GTn6W+Jn/ACT3VP8Atl/6NSuE/aRm2+DdJg+0wLv1Df5DD94+I3G5Tn7q7sHg8uvI6HJfxX6L9Tuqf7jT/wAc/wAqZ4x8O9U8WaDrkureFNJn1GaOIw3EaWbzpsfkBtnK8oCMEfd9Mivoe3f4k+K/hrdidLHw7r9zKBAdrr/oxVSTkM7RScsMkZGMYU4YcP8Asy/8zT/26f8AtavoCtThPiC51/xZomualHLrmq22p+b5N66Xz73ePKYZ1b5tvIHJ9q6TxR4p+I3j3S3v7uz1IaDsZyljaSLabVxuLMM7gGTOXY7SDjFc347/AOSh+Jf+wrdf+jWr7T0nTYdG0ax0u3aRoLK3jt42kILFUUKCcADOB6CgDxv4FeKtYuPD3ifUfEmp31zplhsmS5uy8uzCO0oDHLHCqh2jOMjA+bnzTW/iD43+JPiD+zNPnu4orx3ht9LspCilGGCrsMbxtGWL8D5jhRkV738bf+SQ67/27/8ApRHXkn7OugLqHjG+1qVY2TS7cLHlmDLLLlQwA4I2LKDn+8OO4AI/+GcfGH/QS0P/AL/zf/Gqj0Txp4v+EHjH+xfEr3d7pqIkTWzzNIvkjhJLYtwABnA4BwVIUjK/UdfPH7S9jbx6j4dv1jxdTRTwyPuPzIhQqMdODI/5+woA7v4la940bw1pGqfD2KS4tLhPtE88FussvlsE8oLG4JIbeScKSNozgZz4pY/HXx9aXkc82qQXsa5zBPaRhHyCOSiq3HXgjp6cV738GJ5rn4SaC88skrhJUDOxYhVmdVHPYKAAOwAFfOGrabDrPxsvtLuGkWC98RyW8jRkBgr3JUkZBGcH0NAGpfat8Sfi3eSfZra+msZsQtbWe+KxUoA+GLNs3Zw3zsTkqB/CKk8NfFXxl4E8QTxa8dS1BCgWfT9UmkWRDjKspcEoeQemGB6dCPq+CCG1t4re3ijhgiQJHHGoVUUDAAA4AA4xXyx+0FqUN98TTbxLIHsLKK3lLAYLEtLleem2RRzjkH6kANS8ffEb4layo8PRalawQuI1g0h5EWPex2maQEDOBjc21flJAX5q5ez1Txp8LtZureF7vR7yVNsscsSssqhiAwDAqwyGAcZ74OCa+k/gton9ifC/S99v5Nxfbr2X592/efkbqQMxiPgY9xnNef8A7TX/ADK3/b3/AO0aANTQviI/xB0PRJL2GCDU7LxHaJPHAGCFG3lHG7OM4YY3H7meMgV7ZXyn8H/4/wDsPaV/7cV9WVlD45fI7q/+60f+3vzCiiitThPkfxR4t8aab4dtdEE0ln4dvbd1g8qNR9pXzGMmZOWzuJUqCPlxkYbLcBY2F5qd5HZ2FpPd3UmdkMEZkdsAk4UcnABP4V9F+IP+TX7n/gP/AKWivGPhjNfQfE3w6+nCQzm9RG2R7z5THbLxg8eWXyewyeMZrKh/Cj6I7s0/36t/jl+bPSPB3iX40f8ACQaVZXGn6lcWYdY5E1Ow8mMxgYJecx7gQOd2SSQOGJ2nq/2jv+Seaf8A9hWP/wBFS17BXj/7R3/JPNP/AOwrH/6KlrU4Txz4Z+PZvA9xqK2Ggx6nqmopHBaOWO6NsthQoUs4ZimVBXOwe2DVZvitOlxeamPFywK/2qVnjuI4oyrb9+MBUCkBhgALgYxiuw/Zs02aXxVrOqK0fkW9kLd1JO4tI4ZSOMYxE2ee469vpOgDw/4P/GDVPE2uL4c8RtBLcSxO9rdpHseV1yxRgo2/cyQQFxswclq9wr4g8Cf8lD8Nf9hW1/8ARq19F/H/AFuHTvhu+nN5bT6pcRxIpkCsqowkZwvVgCiqemN457EA4Dxz8f8AV7nVJLXwhJHZ6fE4C3jwB5Z8ZBO1wQqHIwNu75QSRkqMebx18ZdD0sS3jazbWduioZ7vSVwo4UbpHjySTgZJySe5Ncn8PfFNv4M8aWOt3Vj9shh3IyqxDxhgVLpyAWAJ4bg5I4OGHteq/HnwDrml3Gmanoms3NncJslieCLDD/v7kEHBBHIIBGCKAMnwp8apvEWiazoniiW0hu5NNdLK4jiKCeQRsGDndgO3ylQAoJyOpUH2zwt/yKGi/wDXhB/6LWvh/Tf+Qpaf9dk/9CFfcHhb/kUNF/68IP8A0WtZf8vfkd3/ADA/9v8A6GtRRRWpwnyv8RbzxVoes6jqml6xqVnpc+rXVuy2lzLGqzK27LbcLllYY5ydjelcZB408bXVxFb2/iXxBNPK4SOOO/mZnYnAAAbJJPGK+il8GWPjrw14q0e+kkhI8QXM1vOnJhlAADYzhhhiCD1BOCDgj5dnhvtC1mWBzJa6hYXBRjHJ80UqNjhlPUMOoPbisqPwI7sy/wB6l8vyR9Z+PLfxynwxgTQ73fr8MUf9oPagb7hfLKy+T8vDFiGG0K3Hy84B+cNM+KnjnSPN+zeJr6TzcbvtbC5xjOMeaG29e2M8Z6CvqfwT4s/4SX4f2PiXUUgsvMike4O/EaeWzKzZP3V+QtyeAepxmvkDxZfW+p+Mtcv7OTzLW61C4mhfaRuRpGKnB5GQR1rU4T6X+CL+MbnwvLe+J7yeezn2HTRdHdMUyxZ2YjcVbcu3cTwvAC4z558XvGPjTwt8VJWstYu7S0FvG9lCrq0TRlNrFo+VY7/M5cE8DHAWu/8AgT4zbxJ4OOkXEcn2vREjgMp27ZIm3eVjAGCqptPH8IOSScecfEAw/Ej49WehWccjQW7pp08kbhWKxszzsN4ABUGQd87MjOQKAOv0Px54/wBS+C1zrVrpsmpa0b1raGeO2w3kkD98IwMSFWJT5RjjJB2tnyS6+MHj688jzfEc6+TKsy+TFHFlh0DbFG5eeVbKnuDX1/Y2FnplnHZ2FpBaWsedkMEYjRckk4UcDJJP418efFrTYdK+KniC3gaRke4FwS5BO6VFlYcAcbnIHtjr1oA+mx438n4VQ+Mja/2hINPS5lgsTkeYQA65ydqq27cTkqFbOSMV4pP+0P42tbiW3uNI0aGeJykkcltMrIwOCCDJkEHjFe9+BP8Aknnhr/sFWv8A6KWvmj4631xd/FfUoZ5N8dpFBDANoGxDGshHHX5nY8+vpigDY8SftDeJdT8tNDtYNFjXBZvluZGPORl12hTkcbc5HXBxVOz+J/xP8DJBFrEV3JA6OsMeuWb5Y7gSwc7XYjOOWIAbGOmPX/gRpVjZfDCwv7e2jju795Xuph96UpK6Lk+gUcDpyT1JJy/2jv8Aknmn/wDYVj/9FS0Adh8N/HVv488LxX2YI9Si/d31rExPlPk4ODztYDcOvcZJU10Gt63p3hzR59W1a4+z2MG3zJdjPt3MFHCgk8kDgV4f+zL/AMzT/wBun/tasz9o3xDNc+JdP8PxXMbWlpbi4ljjckiZyRhxnGQgUrkZAkPZqAKfiH48eLtb1wW/hYfYLVpfLtoUtkmnn3YC7twYbieiqB97GWwDR/wsn4xaD/xMtWs75rGH/WC/0fyoeflG5lRCOSMYYc469K6f9nPwxpcuk3vieWDzdTju3tIXfkQoI0JKjsx3kE+gwMZbPvFAHwZq2pTazrN9qlwsaz3txJcSLGCFDOxYgZJOMn1NfddhY2+madbWFnH5draxJDCm4naigBRk8nAA618QeLLG30zxlrlhZx+Xa2uoXEMKbidqLIwUZPJwAOtfa/8AxLvDPh7/AJ9tM0y0/wBp/KhjT8WOFX3Jx3oA0KK8nf8AaG8FJLdIItVdYc+W6264uPmC/Jl8jIJb5gvAPfANjRvj54K1fUUs5WvtN34CzX0SrGWJAALIzbeucthQAckUAcn8Ufix438M+IL7SrPTY9NsWcLZ301qWklChCzIxZo2BJ/u5AYAgN05jwr8fPEmnapLN4jnk1ezNu6pAkcMJWXgo2VQHGRtPoGJwSAD7H8bf+SQ67/27/8ApRHXzJ4A0BfFHjzRtHlWNoJ7gNOjsyh4kBeRcryCVVgMY5I5HWgDpL746+Pru8knh1SCyjbGIILSMomABwXVm568k9fTivb/AAL8Uota+HV34p8SrBp8djdm2uJIEdkOdm1go3MP9Yq456Z4BwNTUvhL4D1W4We48NWiOqBALVnt1xknlY2UE89cZ6egrxT43anp2kf2b8PtAh+z6ZpWbmaLcz4mkyyrlwW4V2bIYg+bj+GgCxr/AMcvF3iXXIbPwXaz2UY3iOGO3S5nucZO4qVbbhRnaucfNlmGMZ83xY+K3hq8tp9cE6Rvu2QalpawpNgYPIRGONwPB9M8cGT4J+LfBfhC4v7vXZru21SZDHHcmNngEOUOwBMtvLAkkjGEGCMkHp/iN8W/AvivwdcaMIdSuJ7i3WeCRLZB9muBkqrFzkEH5WKAjazAE5oA9I+HPxGsfiHpdzPBayWd5aOFubZm3hA2djK+AGBCnsCCDxjBPkGrftD+KrXWb63t9I02GCK4kSOO8tpVnRQxAEgEmA4HBHrmsP4A6n9g+KEFt5Pmf2haTW27djy8AS7sY5/1WMcfez2weo/aXsbePUfDt+seLqaKeGR9x+ZEKFRjpwZH/P2FAFOP9pPXhpc0cuh6a2oF8xTqzrEq8cNHklj97kOOo445w5vjP8SreW21Wa58uwupWkt43sEEEqq3zIrFdzKPunDbh655rc/Zy0Cx1DxBqmtXCyNd6Wka2uGwqmUSKzEdztXA7fMeM4I9X+M8ixfCTXmeGOYFIl2uWABMyAN8pByCcjtkDIIyCAZ/wp+Ky+Pkn0/ULeO21q3QzMsCt5UsW4DcuSSpBZQQT3BBOSFw/jP8Un0D7f4QttK82S+0/a93MzKqCTcrBV2jf8vRg2AxwclSK8g+EOs/2J8UNFlZ5xDcymzkSE/f80FFDDIyocox/wB3OCQK93+OPhbS9U8CahrclnAdWsIkFvdPL5ZVDKu5SSQG4LYU55bCjLcgHypBPNa3EVxbyyQzxOHjkjYqyMDkEEcgg85r2CD9pDxUtxE1xpWjSQBwZEjjlRmXPIDFyAcd8HHoa8jsLG41PUbaws4/MurqVIYU3AbnYgKMngZJHWvsPTfhL4D0q4ae38NWjuyFCLpnuFxkHhZGYA8dcZ6+poAn+Gf/ACT3S/8Atr/6Neutrkvhn/yT3S/+2v8A6NeutrKh/Cj6I7s0/wB+rf45fmwrkvD/APyULxj/ANuX/oo11tfMfxy1m8tPE2r6RC+y1v5beW4wSC/lQrtU4OCuXyQQeVU8YoqfFH1/RhhP4Nf/AAL/ANLgM8T/ALQfiXULyaPw+sGl2IlzBI0KyTsgGMPu3JyecBeOBk4JPL2vjj4kr5/iW31fXJLWKVhLcEPJaRu3G0qQYh98YXHGVwBxVP4aeHpvEvxB0eyS2jngjuEuLpZULR+ShDOH4IwQNozwSwB619p1qcJ4H8MPjjfalrNroPiry5XunENtfRRbWMrMdqyKvGDkKCoGMDIOSw7z42/8kh13/t3/APSiOvkzVtNm0bWb7S7ho2nsriS3kaMkqWRipIyAcZHoK+q/i99s/wCFIan/AGj5H27yrX7R9nz5fmedFu2Z525zjPOKAPmz4fCFviN4bE8kiJ/aduQUQMd3mDaMEjgtgE9gScHGD9t18UfDi+uNP+JXhya1k8uRtQhhJ2g5SRhG459VZh7Z45r2/wDaE8TpB4LstIs59/8Aad24laPa6FID86E9QwkKdP7jAkdCAcx40/aD1SbUbqz8JLBb2C4SO+mh3TOQcl1VvlVT0AZScc8E4Xl5vi98T7eztrybVp47W63fZ5n06EJLtOG2sY8Ng8HHSrHwL0/w/deOUu9a1GCC6tNrWFpOFC3MrEqCC3G5SVKqPmLEEfdNfT99Y6X4m0OS1uo4L/TL6IEgNuSVDgqysPwIYHjgg0AeN/DD44tqdxa6D4q8yTULm4ENtfRRKFkLk7VkVcbTuwoKjByMgYLHY+LnxZ1LwHrOn6XpFpaTTy25uJ2u42ZQpYqgXa6nOUfOR/dx3r508V6N/wAI94t1bSAk6R2l3JFF54w7RhjsY8DOV2nIGDnI4r3dY9C1v9m+PXNf02xN3baU9nb3QgPmI0UjRQAOMsMuEzztyzZAUkUAc5pX7SGvLqludY0rTX0/ficWkbrLt9VLORkdcHrjGRnI9b+Kvje88BeEo9TsLWC4uprtLZBOTsTKsxYgEE8IRjI657YPxxX3f/xLvE3h7/n50zU7T/aTzYZE/BhlW9iM9qAPnD/ho7xh/wBA3Q/+/E3/AMdr1P4K+KtT8YeGrzUdZhjkv4bj7Kb9Yo0M6Ab1RtvOUMjfwgYcYyd1eAfFjSrHQ/iVqmmaZbR21nbpbpFEnRR5Ef4kk5JJ5JJJyTXs/wCz/p923w5d/Pu7JG1h51ZI0xcRiONSvzqcoWVgSuDlSARg0Adh41+KPhzwHcQ2mqNdzXkqLIttaw7mEZLDeSxVcZQjGc9OMc14ZrH7QPi681xLzS/I06wTZiwaNJg+OW3uVDHPI+XbgYxzyeH8aeJP+Ep8UXWoxR+RYriCxtgu1YLdBtjQLkheBkheNxbHWvd/hZ8KvDV58Oo7nWrWx1abVts5mjLZgQY2xK4IKsCG3bdvJKnIWgDyDW/iv4l8S+F59C1z7DfxyyrKt1LbKs0JUjGwphR0IztJw7DODxx9hY3Gp6jbWFnH5l1dSpDCm4Dc7EBRk8DJI617B8V/AOheH/AOka5Y6DPoepS3a29zZveG4A3I5OW3MDgx8EEZDcjPA8XoA+wvhbo1nomna1Z2qZ+zanJZiZwPMeOIKEDMAM4yx9MscAZrva5LwJ/zMv8A2Hrr/wBlrrayo/AjuzL/AHqXy/JBRRRWpwhRRRQAUUUUAFFFFABXzn8Y73WbbWfEFvpd0YoLm4tUuo4ZHWeVfszDGF4aLBYOD3MdfRlZ11oGjXtw9xd6RYTzvjdJLbI7NgYGSRnoAKzqRbs49P8AJnXhK1OmqkaqdpK2m/xRfX0Phr+zb7/nyuP+/Tf4V658E/FfiHSfEFtod7drB4bCSySrfFY1hJGQY2Yg5L4+UEj5mO3OWH0J/wAIt4e/6AOl/wDgHH/hR/wi3h7/AKAOl/8AgHH/AIUv3vkX/sP9/wDAw/GHxA0/Q/Cd/qOj3en6lqESKILVLlXLszBc7VOSFzuIHUKeR1Hyfr+o+JfFOqNqWtG8u7soE3tDtCqOgVVACjqcADkk9Sa+zP8AhFvD3/QB0v8A8A4/8KP+EW8Pf9AHS/8AwDj/AMKP3vkH+w/3/wAD5D8GeJ/FHgXVHvtHt5SJU2TW88LtFMOcbgMHIJyCCCOR0JB3PH/xJ8V+Ot9n9iuNP0Zth+wRKW3MvOXfaC3JzjhRheMjJ+oP+EW8Pf8AQB0v/wAA4/8ACj/hFvD3/QB0v/wDj/wo/e+Qf7D/AH/wPi/RLrX/AA5rEGraTFcW99Bu8uX7Pv27lKnhgQeCRyK9w8Y/FvxPD4L8PXGkRWVvq2oxSNfLABcvalSmwgZITeCx2uCRnHVSa9g/4Rbw9/0AdL/8A4/8KP8AhFvD3/QB0v8A8A4/8KP3vkH+w/3/AMD4nsI9X0zUba/s7a4jurWVJoX8gna6kFTgjBwQOtfWngb4iweIfDUd/rx0/R7wuU8h71AZAAAZNjEMgLbsK2TgA5IINdJ/wi3h7/oA6X/4Bx/4Uf8ACLeHv+gDpf8A4Bx/4UfvfIP9h/v/AIHyX4z8HS+DvEqPoGqLqNmX86zu7KcPLCVIID7DlXU4wwwD1GDkL6Jo37QHiO005IdX8KnULpMD7RC7W+8AAZZdjDcTkkjA54AxXuH/AAi3h7/oA6X/AOAcf+FH/CLeHv8AoA6X/wCAcf8AhR+98g/2H+/+B8f+MfEHiPxxrn9ratZFZliWGOOC3ZUjQZOBnJPJY8k9fTAGp4E1rW9EmWzXT7m6tkvIdQitNrhpJ48gInUAuGwcKSdq+mK+rP8AhFvD3/QB0v8A8A4/8KdF4b0KCZJodF06OWNgyOlqgZSOQQQODUyjUkrOxrQrYOjNVIqTa9OxqUUUVueYFcl4E/5mX/sPXX/stdbXJP4Bt/tl3cW+va9afap3uJI7W8Eab2OSQAv4fgKympcyaV7HdhZ0vZVKdSXLzW6X2ZyH7QHhSbXPB0Gs2okefR3Z3jXJ3QvtDnABOVKo2cgBQ5PauI/Z18LrfeIL7xJcRybNOQQ2pKMFMsgIYhs4JVOCuD/rQeMDPs3/AAgn/U1+KP8AwY//AGNV7H4bWemWcdnYeIPENpax52QwXgjRckk4ULgZJJ/Gjnn/AC/iHsML/wA/v/JWWfidpsOq/DLxFbztIqJZPcAoQDuiHmqOQeNyAH2z06182fB7wb/wl/jm3+0w+Zpmn4urvcuUfB+SM5UqdzdVOMqr46V9Jf8ACCf9TX4o/wDBj/8AY1T034Xabo1u1vpeta7YwM5do7W6WJS2AMkKgGcADPsKOef8v4h7DC/8/v8AyVnc15v8SPi9pfgbzdNtk+3a6YtyQj/VwE42mU5yMg7go5IAztDBq2/+EE/6mvxR/wCDH/7GsvUvhBoOs3C3Gqahq19OqBFkupklYLknALITjJJx7mjnn/L+Iewwv/P7/wAlZ8o6Fqf9ieIdM1byfO+w3cVz5W7bv2OG25wcZxjODX1vb/FXw/d/D+78ZQRXz2FpKIZ4PKUTI5ZVAwW2n76tw3Q+uRWV/wAKK8H/APT3/wCQf/jdXI/hBoMOlzaXFqGrJp8z75bRZkETtxyybME/KvJHYelHPP8Al/EPYYX/AJ/f+Ss+efir43s/Hvi2PU7C1nt7WG0S2QTkb3wzMWIBIHLkYyeme+B7v8EfGln4g8F2mjS3e7V9Li8qWF1Ckwg4jZcfeULsUnqCOeoJf/worwf/ANPf/kH/AON1c034QaDo1w1xpeoatYzshRpLWZImK5BwSqA4yAcewo55/wAv4h7DC/8AP7/yVnzj468Hap8OvFpiHnxW/mmbTL1H5dFbKkOAMSL8ucAYPI4IJ9z8L/H/AML6jpaHxBJJpWoIiiUCB5IpG5yYygYgcA4bGN2AWwTXR6l8LtN1m3W31TWtdvoFcOsd1dLKobBGQGQjOCRn3NZf/CivB/8A09/+Qf8A43Rzz/l/EPYYX/n9/wCSs5j4h/Gy01DS10TwJdXdxqd66R/aYbZ1Kq24FI92H80naAQp4Y4O7GOs8N6fpfwT+Gsja1qPmMZTPOUH+snZQBFCpwTwgAz6FjtGds+m/CDQdGuGuNL1DVrGdkKNJazJExXIOCVQHGQDj2FWNS+F2m6zbrb6prWu30CuHWO6ullUNgjIDIRnBIz7mjnn/L+Iewwv/P7/AMlZ8k67qf8AbfiHU9W8nyft13Lc+Vu3bN7ltucDOM4zgV9l+GvHnhzxXo0+qabqMYgtUDXaz/u2tvl3HzM8AAZ+YEr8rYJwa5b/AIUV4P8A+nv/AMg//G6sQ/Bjw3b2dzZw3Wpx2t1t+0QpJGEl2nK7lCYbB5GelHPP+X8Q9hhf+f3/AJKzwn4z+LdI8Y+NYb3RZpJ7SCyjt/OaMoHYM7kqGwcfOByByD2wTr+BfFOjWnwi1nw1PebNXu9Q86C38pzvQCEk7gNo+43U9vpXq/8Aworwf/09/wDkH/43T4/gd4ThkDxPfI46MpiBH/kOpqe0lBx5d/M6MIsJQxFOs6t+Vp/C+juelUUUVueSFfNfxC0t7n4IeF9RjsvN+x3cyyXAdswo8jjBUDbtZlTLEjBCgZ3HH0pXm1v4R8WR+Ej4ZuIfDN5pjIyOlyZyWDMW6gDBBOQRgjAI5Gaxm2pqVuj/AEPRw0Y1MLUpOSTcovV20Smn+aPEfhF8R7P4f6jqf9pW081jfRJn7MgaQSITt+8yjbh3z1OdvvX0H4H+J+g+Pri8t9Lju4J7VFdo7sIrOpJGVCuxIBABPbcvrXl99+zzfXd5JPDLpllG2MQQXExRMADgujNz15J6+nFb6/CK9g0fU9NtNH8L20eoxLFNKk920gCsGXazltuGCtjoSoyCBin7XyZH1H/p5D7zy341+KdG8XeMrO/0O8+12senpCz+U8eHEkhIw4B6MPzr2P4HeJNFl+HmkaKuqWg1SJ50azaULKT5jyfKp5YbWByMjr6HHD/8M46n/wBBK0/7/t/8arc8JfBjXfB3iCLWrKfSbi7hRli+1TSlULDaWAVVydpI5yOTxnBB7XyYfUf+nkPvOk+N/iSx0z4eanpn9qRwapepEkNukuJXQyDfwOQhVZASeDyO+D5J+zzfW9p8SpIZ5Nkl3p8sMA2k73DJIRx0+VGPPp64ru/F3wd1rxlrDatenRbW+kx50tpPOPNwqquQ4YDAUD5QOpzmuf8A+GcdT/6CVp/3/b/41R7XyYfUf+nkPvOs/aJ0T7d4GtNWjt982m3Y3y78eXDINrcZ5y4iHQkfTNV/gJ460u78OW3g+U/Z9Ts/MaEO3F0jO0hK/wC0u45X0GRn5tvbzwePbq3lt7iLwpNBKhSSORZ2V1IwQQeCCOMV5Zffs8313eSTwy6ZZRtjEEFxMUTAA4Lozc9eSevpxR7XyYfUf+nkPvPbPE/ifS/COhzavq8/lW8fCqvLyueiIO7HB/IkkAEj5o+FujXnj74tNrlynlQ2122q3bwgqgkL70RSQ3V+xOSqvg5FdXffs/6/qd5JeX+uR3d1JjfNPdySO2AAMsY8nAAH4V6XpWleNND0u30zTLbwpbWdumyKJPtGFH8ySckk8kkk5Jo9r5MPqP8A08h95B8bf+SQ67/27/8ApRHXjn7Os8MPxGuklljR5tMlSJWYAu3mRthfU7VY4HYE9q92/wCLh/8AUr/+TFeWar+z/qOp6pcXySaTYid9/wBntJZFiQ99oZCQM84zgZwMDAB7XyYfUf8Ap5D7z3i+v7PTLOS8v7uC0tY8b5p5BGi5IAyx4GSQPxr4c8S6lDrPirV9Ut1kWC9vZriNZAAwV3LAHBIzg+pr2e+/Z/1/U7yS8v8AXI7u6kxvmnu5JHbAAGWMeTgAD8Kr/wDDOOp/9BK0/wC/7f8Axqj2vkw+o/8ATyH3nqfwu1KHWdG1fVLdZFgvdWmuI1kADBXVGAOCRnB9TXc1w3w28BS+CbO8+1XCS3VwIov3TlkEcYbb1UHdl3z26dOa7mnSTUFcnMJxniZOLutPyCiiitDjPhvxTY3FpqqzTx7I7uITQHcDvQExk8dPmRhz6emK+mPgf4th8ReA7fTnmkfUNIRbecNGFAjy3klSOCNi7fXKHPUE4SaZpeq+BbDQPEvgzxHcT2nmmK4t7Jg0LOzHKNkHoQcEFSVGQcCvKJvhh4qsdUE2j2erFIXV4Ll7GWCUMMHOF3bSG6EMegPHQc1KrBU4pvoj2swwGIqYurOEbpyk1qtm35n1hrOuaX4e057/AFe/gsrVcjfM+NxAJ2qOrNgHCjJOOBXxp8P9VbRfiDoF+tzHaol7Gks0m0KsTnZJktwBsZue3XjrXb2Pw0vNTvI9Q8Xf8Jfd3UmftKwaaZHbAKpieR8nAC9U7YHY1lt8JNS33oWDXSiIDZk6MwMzbTkSDf8AuxuwMjfxzgdK09tDucf9m4r+X8V/meyftAarfaZ8OUSxuZIBe3qWtxs4LxGORimeoBKjOOoyDwSDyf7PGh+GtS06/v7mwguddsbsFXmRm8mJgpjZQflDb0fDD5hjqARnUuNC0u7+F9p4Kn0Dxe62mZYL3+zcMk5LMX2BsFcuw2k/dOM5w1eRwfDPxta3EVxb6ZfwzxOHjkjt5lZGByCCEyCDzmj20O4f2biv5fxX+Z7J+0Vr7af4OsdFiaRX1S4LSYVSrRRYYqSeQd7REY/unnseY/Zx1bTdPuNft73ULS2nu3tEto5plRpmzKMICcscsBgeo9arW/w6t7/Tru48S2vjbUNfuogBd/YSyQOAuD8zbpMbSuSRlT0UgEcXB8KvFTXES3GlX8cBcCR47KV2Vc8kKVAJx2yM+oo9tDuH9m4r+X8V/mfYk88Nrby3FxLHDBEheSSRgqooGSSTwABzmviTwJ/yUPw1/wBhW1/9GrXr/wAVl8ZeOL2CDR9O1220VbcLLZz2skYkl3lizBAQwwI8ZPBBwBkk+Z/8Kr8Yf9Aa7/8AAWb/AOIo9tDuH9m4r+X8V/mfZlc/47/5J54l/wCwVdf+imrgvhxrfiTw54euLPxRpfijVL57tpUm+zyT7YyiALukII5DHHTn3rpNV8TWOuaXcaZqfgvxJc2dwmyWJ9O4Yf8AfWQQcEEcggEYIo9tDuH9m4r+X8V/mfPPwS/5K9oX/bx/6TyV9f18d33wo8SR3ki2Gm6nPajGySfT5InbgZyo3Ac5/iPrx0rrPDOk/EG28WaTf+I5fFd3p9lcfaDHE1xI5YKwAAcYwc7W6ZVmHej20O4f2biv5fxX+ZxHxU0z+yPih4htvO83fdm53bduPOAl24yem/Ge+M8dK+r/AATrml+IPCWn3ekX897bpEsLSXT7pw6qARN/006E+ucjIIJ8z+JGjaX4506Wa38IeIbLXV+eK9Gl480gAbJSDllwAAeSuBjjKnyvTfAPxB0a4a40u21axnZCjSWqXETFcg4JVQcZAOPYUe2h3D+zcV/L+K/zPqjxP4n0vwjoc2r6vP5VvHwqry8rnoiDuxwfyJJABI83+GPxV1TxLpPi3VvEEUH2fSIkuljs4tpVPLcuo3Nz/qsjJ6secYA8rTwB4w1rXLW48VxeIbu1XCTTLBNPOIxk7U8wYGST1OBknB6H1tWsdJ+HOo+FNA8H+JIBcWU0CyTWHMkrxlfMkYHJJOMkDgAAAAAUe2h3D+zcV/L+K/zNHwH8RNH+K1nqWm3+jQQyQbHexuZEuEmjzwwBUZ2sBn5cDKc5PHH/ABt+G/hbTPC8viPTYYNKvopY08iHCR3W4hdqpkBWABf5RyA+Qc5Hmdj8OvHmmXkd5YWOp2l1HnZNBFPG65BBwwXIyCR+NbHifQvij4x+y/2/b3d59l3+T/xL2j27sbvuRDOdq9fSj20O4f2biv5fxX+Z6B+zZqUMvhXWdLVZPPt70XDsQNpWRAqgc5zmJs8dx17eWfG3/kr2u/8Abv8A+k8dexfDWT/hAvCS6VJ4f8UXN1LKbm5ddOwgkZVUqnOdoCgZPJ5PGcDjPit4Y1Lxr4lg1nRfD2rW7tbiK5S405oy7KTtfcu7cSpC8gYCLyex7aHcP7NxX8v4r/M9Y+EOs/238L9FlZ4DNbRGzkSE/c8olFDDJwxQIx/3s4AIq34f/wCSheMf+3L/ANFGvmD/AIVX4w/6A13/AOAs3/xFe5fAzQNU8P6bqlvqljcWznytpliZA/MhONwGcZH51MpxlKKj3/RnRTwtWhQryqK14pbr+eD/AEPWqKKK3PJOS+Jn/JPdU/7Zf+jUryz9pr/mVv8At7/9o16n8TP+Se6p/wBsv/RqV8ueJdQ8Z+MLiCfXriS8eBCkQJiRUBOThVwMnjJxk4HoKwc4xqvmdtF+p6sMLXr4GHsYOVpzvZN9Idj1D9mX/maf+3T/ANrV9AV8T6A3ivwtqi6losklpdhCm9XjYMp6hlbIYdDgg8gHqBXsHjT4ja9c+BdDg8PaxGNWkt1TVysRScOYgG2MVCKN2/JUgg7dvGar29L+ZfeYf2Xjv+fM/wDwF/5HjHjv/kofiX/sK3X/AKNavsfwnfXGp+DdDv7yTzLq60+3mmfaBudo1LHA4GST0r4r/wCEf1T/AJ9f/Ii/417B8H/GWsaBqMth4o1WdNCS0It0m/fbJQYwqqQGZVCBsL90emaPb0v5l94f2Xjv+fM//AX/AJHqnxgsbjUPhRr8NrH5kixJMRuAwkciSOefRVY++OOa8M/Z91KGx+Jot5VkL39lLbxFQMBgVly3PTbGw4zyR9R71P8AEXwVdW8tvcalHNBKhSSOS0lZXUjBBBTBBHGK+dPGPhTTNL1z+0vBGseda+askEA82Ke1bk/K7KAVUgYbdu5HBwWJ7el/MvvD+y8d/wA+Z/8AgL/yPryvmz9o3X7HUPEGl6LbtI13paSNdZXCqZRGyqD3O1cnt8w5zkBJvjD8QLjw9cabLp1iLqWIQi/iYxyKCjKz4V8CQkqwYYCkH5TkYw/AnhrSn8S22u+NdXtJYN7zzWU0c88s0uTjzCF2kE/OfmbOACOTg9vS/mX3h/ZeO/58z/8AAX/ke8fB+xuNP+FGgQ3UflyNE8wG4HKSSPIh49VZT7Z55r5gv7nUX+KdzdboNI1NtbeTNxKrR2c3nk/O+CpVG6tgjAzivfviH8USnhpV8E6jHJqj3CK7NbnMcWGJZfMAXOQo5zwx47j5s/4R/VP+fX/yIv8AjR7el/MvvD+y8d/z5n/4C/8AI+7K+RPjjBND8W9XeWKREmSB4mZSA6+Si5X1G5WGR3BHavQ/DXxZ8Ujwnq66xJpra1bpG2nNcQNi6JZjIr+UdoIXaF+4MkZJ5NeN63F4l8R6xPq2rL9ovp9vmS5jTdtUKOFwBwAOBR7el/MvvD+y8d/z5n/4C/8AI+sPhjqUOq/DLw7cQLIqJZJbkOADuiHlMeCeNyEj2x06V5X+01/zK3/b3/7RqP4NeN7zw3BPofiWXydIii32T7Q3lPvJZMIpZt28nJPGzHevN/FmqeMfGuorea03m+VuEEKMiRwqxyQqg/QZOWIAyTgUe3pfzL7w/svHf8+Z/wDgL/yOg+D/APH/ANh7Sv8A24r6sr5H+GcGq6d4t060mXy7K4v7WR1yp3SJJhDkcjAd/bnntX1xSpyUpycXfY1xtGpRw9GFWLi/e0as9/MKKKK2PMPE/EH/ACa/c/8AAf8A0tFeGeBP+Sh+Gv8AsK2v/o1a+ovB+jWfiH4QQaRfpvtbuKeJ8AErmV8MuQQGBwQccEA18w6/4R8TeBtUZr+yu7U21wFh1CFWETOPmVo5cAZwMjoRg5AIIGVD+FH0R3Zp/v1b/HL82fbdeP8A7R3/ACTzT/8AsKx/+ipa5TQNX+OviDVFsVmu9PGwu1xqOlxwRKB7mHJJOAAATznoCRy/xb+IWveJNUufD9/bR2mn2N68kEZtXhlkXkRPIJPmB2NnGF+/yOmNThOz/Zl/5mn/ALdP/a1e+TwQ3VvLb3EUc0EqFJI5FDK6kYIIPBBHGK+SPhzq/j7wmlzqHhrw5d6hZ6ggVi2nSzROUYgMrJg5BLjg45ORkDHr/wAYviB4v8D3FuNIsbQaXd25QX8sLO0dxlsgfNtBC7WAZSDz1wQAD588Cf8AJQ/DX/YVtf8A0atex/tMwTNb+GrhYpDAj3KPIFO1WYRFQT0BIViB32n0rwSwvrjTNRtr+zk8u6tZUmhfaDtdSCpweDggda+r7Wa4+MHwXnFza/2fdahEyIRIPLeWJ8q4PzERl0wQRuADAZ4YgHmHwR8KeCfGGnalZa5p32nV7aUSqWuZI90DAAbVRxnawOTjjevPPHq//Ckvh5/0L3/k7cf/AByvmhE8S/C7xpa3FxZ/YtWs8SokwWRHRgQeQSGUgspKnjnkEcekf8NKax/Y/lf8I/Y/2n/z8ec/k/e/55fe+7x9/rz7UAdb8StO+HnhKeyln8LxpqmoQT29hJZQqkUTgABnQMq5BlBDbWIxn+EV6Z4W/wCRQ0X/AK8IP/Ra18Xpfap4k8Vrf3kk99fzzedM+3cxC8scDoqqp6cKq9gK+0PC3/IoaL/14Qf+i1rL/l78ju/5gf8At/8AQ1qKKK1OE5LwJ/zMv/Yeuv8A2WvEfj74GbR/EA8UWcUa6fqThJwpUbLnBJ+UAcMq7s8ktvJxkZ9u8Cf8zL/2Hrr/ANlq7408J2fjXwvdaLeP5Xm4eGcIHaGRTlWAP4g4wSpYZGc1lR+BHdmX+9S+X5I+TNN8cTaf8N9Z8GtYxyQajcR3CXIkKtEwZCwIwQwIjUDpjnr0Hplj8Jkv/gDHdRWP/FQy51WNgqvJImDtiUqpYq0WGCZ++RnHIHnfw58EzeKviDb6Le28kcFq7S6jHICrJHGQGQjKsCWITjkbs44NfZdanCfFHgvxzqPgaXU7jTIoDdXlp9njmkjVjA25TvGRk4APy5Ck7SwbaBXr/wCzt4Pt10678WXltuunlNvZNLER5aAfO6EnB3FimQONjDPJFeUa14I/sz4qnwaLrbHJqEVtFOR5hWOUqUZhhcsFdcgYGQccc19h6VpVjoel2+maZbR21nbpsiiToo/mSTkknkkknJNAFyvkD42/8le13/t3/wDSeOvoP4reO77wD4agv9P06O5nuLgQLLOf3URwW+ZQwZiQrYxxwSSMAN8karqt9rmqXGp6ncyXN5cPvllfqx/kABgADgAADAFAH2n4E/5J54a/7BVr/wCilr54/aG+2f8ACyo/tPkeT/Z8X2Xys7vL3Pnfn+Lfv6cbdvfNdJ8M/jW1h4aOj6xpGpagdLtwYp9Nt1kK2yDGZRkBQg2jf0IIzgjLeP8AjDxLN4w8WX+vT28du926kQoSQiqoRRk9TtUZPGTngdKAPqP4Jf8AJIdC/wC3j/0okrh/2l7G4k07w7frHm1hlnhkfcPldwhUY68iN/y9xVf9nzxvqM8reDZrXz7OCKS5t50Kqbdd2WVgSNyln4IywLd15XnPjn49vtd8QTeGBaSWen6XcZKzJiSeUAgSeybWO0DqG3HqAoB0f7Mv/M0/9un/ALWrkPj9pn2D4oT3PneZ/aFpDc7duPLwDFtznn/VZzx97HbJPgv8QLzwt4hi0P7P9qsdYu4IdjTFfIkZwnmKOQeDyMAnavIxz7n8TPhnY+P9LDoY7bWrdCLW7I4I6+XJjkoT36qTkdSGAMf4Aalpt18N0srRY0vLO4kF4oChnZmLK5wckFcKCf8AnmR0WvVK+KPCnjHxB8OtcuJbD91NzDdWV2jbHK5GHTIIZTnHII5HQkH0jWf2kdUu9OeHSNCg0+6fI+0TXH2jYCCMqu1RuBwQTkccg5oA87+J2mzaV8TfEVvO0bO969wChJG2U+ao5A52uAffPXrX2HrmjWfiHQ73SL9N9rdxNE+ACVz0ZcggMDgg44IBr4c1WDUrbVLhNYiu4tQL751u1ZZSzfNlt3OTnOT1zmvpf4O/FDWvHFxcaXq2nRs9nbiR9ShBVWOVVVdcYDt87ZBA+U4UYoA88/4Zx8Yf9BLQ/wDv/N/8aqxY/s3eJJLyNb/WdKgtTnfJAZJXXg4wpVQecfxD156V9L0UAef/ABt/5JDrv/bv/wClEdfPnwYnhtvi3oLzyxxIXlQM7BQWaF1Uc9yxAA7kgV2nxe+Lv9rQa14Ls9J8uGO7FvNeTS5Z/KcFtqAfL86DBLH5R0BPHkfhzX77wt4gs9a01oxd2rlk8xdysCCrKR6FSRxg88EHmgD7rr5Q+P2mfYPihPc+d5n9oWkNzt248vAMW3Oef9VnPH3sdsn3v4c/EOH4haXc3cWlXdg9s4SXzCHiZjk4STjcQoUsCBjcOuc1h/F74W/8JxZpqumNs120i8uNHfCXMYJbyznhWySQ3TnDcYKgHGfBj4c+D/FngqbUNY06S9vEvZIWZpZIggCoQq7Hwww2ckA5YjGACfQ/+FJfDz/oXv8AyduP/jlfPngfxvrXwp8QXkVxpchSdFW80+6UwSZAJjYEqSpG7PQghjx0I7jUv2ltSlt1XS/DdpbT7wWe6uWnUrg8BVCEHOOc9jxzwAeh3ureCvBPj620Ox0eDS9d1e0SKC8gsVMC7ncRq6oynlxzgDPy5YBQV4f9pr/mVv8At7/9o1Y+DXhHXdU8UT/EHxSs7TTxeZYyzkBpmkBUyABhtUINqqV2lXBXhRXnHxT+JX/Cw9Rs/I0/7HY2HmiDe+6STeRlmxwvCL8ozg5+Y8YAO/8A2Zf+Zp/7dP8A2tXoHxt/5JDrv/bv/wClEdfPnww+JM3w91SctZx3Wn3zxi8Az5qqm7BjOQM/OTg9cYyvWvQ/jV8UEls9S8DjRZ4ppIrd555p1BhclJtm1dwbA2gkN1JxkAEgHmHwrtbO8+KHh6K+uPIhW7EqvvC5kQF41yf7zqq46nOByRX0f8bf+SQ67/27/wDpRHXyx4V17/hGPFGna19igvfscok8icfK3GOD2YZyrc4YA4OMV9dxXOm/FH4bzvbxyQ2erW80Uf2uFWaJgzIHKhiMqy7hz2HINAHyZ4E/5KH4a/7Ctr/6NWvt+vhCwurzw54htrz7PsvtNu0l8m4QjbJG4O114I5GCODXskP7S2pLpZjn8N2j6hsYCdLlli3c7T5ZBOBxkb+cHkZ4APXvhn/yT3S/+2v/AKNeutrkvhn/AMk90v8A7a/+jXrrayofwo+iO7NP9+rf45fmwr5T+PX/ACUK4/4D/wCioq+rK8a8aeBU8dar41t4hjU7P7JPZMFXLOIG/dEtjCvwDyMEKTnbglT4o+v6MMJ/Br/4F/6XA8t+BV9cWnxX02GCTZHdxTwzjaDvQRtIBz0+ZFPHp6Zr63r4QvrHVPDOuSWt1HPYanYygkBtrxOMFWVh+BDA88EGvZNN/aW1KK3ZdU8N2lzPvJV7W5aBQuBwVYOSc55z3HHHOpwnvep6VZ33lXUul2N7fWeZLI3aD93JwRh9rFMlV+YAkYBwcVx/xt/5JDrv/bv/AOlEdeAXV141+NfiiAC383ytsSrCjJaWSsMlmJztztJJJLNjAzhVHofx1+JljLYXXgrSxHdSu6i/uM5WEo4YRrjq+5RuPRenJztAPGPBcEN1468PW9xFHNBLqdskkcihldTKoIIPBBHGK9I/aGjW78R6RrVvNHJaTW81iMBgwlt53WUEEDjc+Ae+D2wT5XoWp/2J4h0zVvJ877DdxXPlbtu/Y4bbnBxnGM4NfR/ioaX8YPhLqOo+GoJ0ure7N0YGh2SSzxJtKuFDb2MTDbgnnYMjBAAOM/Z0fTbrWdZ0u9so7mcpBf2zSxK6xNCzLuGeVcGYYIHryO/0nXxJ4S8S6l8P/GMWpJbyCe1dobqzlLR716PG46gg8jIOGUEg4xXvfib4/eHbfwubnw5cfatZk8vy7S5tZAsWSC3mHKjgZHysfmI6jJoA8I+I99cah8SvEc11J5ki6hNCDtAwkbGNBx6Kqj3xzzXqFvDcR/sk3bz3XnRyyh4E8sL5KfbFBTI+98wZsn+/joBXmfgzwZrXxH8SvBBJIQX86/1CfLiIMSSzEnLOxzgZyxzyACR9V6r4Ns/+EG1Hw/oEMGmNPp7WULxqB8uG2q7FWJUl3yeW+dyDuOaAPiivv+vgjFxpmo4lg8u6tZcNDcQg7XU8q6OMHBGCrD2Ir3//AIaP0ubTvPn8O3yalDLugt4rz9y4xtJdwAejN8pRhkKeuCoB5x8bf+Sva7/27/8ApPHXs/wEu0l+H9lAda86SKW6VbDy1TyhuRiMkbn27w24HA8/ackLj5k1bUptZ1m+1S4WNZ724kuJFjBChnYsQMknGT6mvXPgD4gs21iTwpqtvBcQzy/2hp/nRB/Kuo15K/KcMUGQxIx5eBy1AHj9/Y3Gmajc2F5H5d1ayvDMm4Ha6khhkcHBB6V9l/DHUodV+GXh24gWRUSyS3IcAHdEPKY8E8bkJHtjp0r5w+L/AIFm8JeMbu4tLORdFvXE1vIkREUTPuJhyFCggq5CjOE2+9anwt+Mf/CEac2i6tZT3mmGVpY5IZMyW+Ryqqx2lSwBwCuCznnOKAPZ/jb/AMkh13/t3/8ASiOvkCvQPih8Tbj4gajDHbxz2ekWufKtXkB8x8t+9YAcMVIGMtt5wfmOfP6APtLwJ/zMv/Yeuv8A2Wutryr4Gaz9v8PX1nO88t8jx3k80p3eZ5yYB3E5LZiYnPqOTzj1WsqPwI7sy/3qXy/JBRRRWpwhRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAcl/wAKz8If9Aj/AMmZf/i6P+FZ+EP+gR/5My//ABddbRWXsKX8q+47v7Ux3/P6f/gT/wAzkv8AhWfhD/oEf+TMv/xdH/Cs/CH/AECP/JmX/wCLrraKPYUv5V9wf2pjv+f0/wDwJ/5nJf8ACs/CH/QI/wDJmX/4uj/hWfhD/oEf+TMv/wAXXW0Uewpfyr7g/tTHf8/p/wDgT/zOS/4Vn4Q/6BH/AJMy/wDxdH/Cs/CH/QI/8mZf/i662ij2FL+VfcH9qY7/AJ/T/wDAn/mcl/wrPwh/0CP/ACZl/wDi6P8AhWfhD/oEf+TMv/xddbRR7Cl/KvuD+1Md/wA/p/8AgT/zOS/4Vn4Q/wCgR/5My/8AxddPa2sNlZwWlumyCCNY41yTtVRgDJ56CpqKqNOEfhVjGti8RXSVablbu2/zCiiirOc5J/ANv9su7i317XrT7VO9xJHa3gjTexySAF/D8BR/wgn/AFNfij/wY/8A2NdbRWXsYdju/tLFfz/gv8jkv+EE/wCpr8Uf+DH/AOxo/wCEE/6mvxR/4Mf/ALGutoo9jDsH9pYr+b8F/kcl/wAIJ/1Nfij/AMGP/wBjR/wgn/U1+KP/AAY//Y11tFHsYdg/tLFfzfgv8jkv+EE/6mvxR/4Mf/sar33w2s9Ts5LO/wDEHiG7tZMb4Z7wSI2CCMqVwcEA/hXa0Uexh2D+0sV/N+C/yOG034Xabo1u1vpeta7YwM5do7W6WJS2AMkKgGcADPsKr6b8INB0a4a40vUNWsZ2Qo0lrMkTFcg4JVAcZAOPYV6DRR7GHYP7SxX834L/ACOS/wCEE/6mvxR/4Mf/ALGj/hBP+pr8Uf8Agx/+xrraKPYw7B/aWK/m/Bf5HJf8IJ/1Nfij/wAGP/2NH/CCf9TX4o/8GP8A9jXW0Uexh2D+0sV/N+C/yPPtS+EGg6zcLcapqGrX06oEWS6mSVguScAshOMknHuaNN+EGg6NcNcaXqGrWM7IUaS1mSJiuQcEqgOMgHHsK9Boo9jDsH9pYr+b8F/kcNqXwu03WbdbfVNa12+gVw6x3V0sqhsEZAZCM4JGfc1c/wCEE/6mvxR/4Mf/ALGutoo9jDsH9pYr+b8F/kcl/wAIJ/1Nfij/AMGP/wBjR/wgn/U1+KP/AAY//Y11tFHsYdg/tLFfzfgv8jz7UvhBoOs3C3Gqahq19OqBFkupklYLknALITjJJx7mqf8Aworwf/09/wDkH/43XptFHsYdg/tLFfzfgv8AI5L/AIQT/qa/FH/gx/8AsaP+EE/6mvxR/wCDH/7Gutoo9jDsH9pYr+b8F/kefal8INB1m4W41TUNWvp1QIsl1MkrBck4BZCcZJOPc1T/AOFFeD/+nv8A8g//ABuvTaKPYw7B/aWK/m/Bf5HJf8IJ/wBTX4o/8GP/ANjVOP4XabDqk2qRa1rqahMmyW7W6USuvHDPsyR8q8E9h6V3NFHsYdg/tLFfzfgv8jz7UvhBoOs3C3Gqahq19OqBFkupklYLknALITjJJx7mo774MeG9TvJLy/utTu7qTG+aeSOR2wABlimTgAD8K9Foo9jDsH9pYr+b8F/keZf8KK8H/wDT3/5B/wDjdbkHw9htbeK3t/EviSGCJAkccd8FVFAwAAFwABxiuxoo9jDsH9pYr+b8F/kedX3wY8N6neSXl/dand3UmN808kcjtgADLFMnAAH4VX/4UV4P/wCnv/yD/wDG69Noo9jDsH9pYr+b8F/kZ2haNb+H9Gt9LtHleCDdtaUgsdzFjnAA6k9q0aKK0SSVkclScqk3Obu3q/UK5K78Kax/wkOo6rpXiT+z/t/leZF9hSX7ibRyx+p6DrXW0UpQUtzShialBtwtqrO6TVrp7NNbpHm3iX4X6h4wt4INe8TR3iQOXiJ0xEZCRg4ZWBweMjODgegrnP8AhnLS/wDoL/8Aks3/AMdr2yio9jHz+9/5m/8AaNbtD/wXD/5E828L/D298PaWi+G/FVpBaToriaDSoXM6nLKTIWJcfMcEk4B44rP1/wCCg8Uao2p6xr0c94yBGlXTxEXA6btjgE44yecADoBXrNFHsY+f3v8AzD+0a3aH/guH/wAieLT/ALPOm3NxLO+rRh5HLsI7IooJOeFWQBR7AADtXUaB8PdY8LaWum6L4qjtLQOX2LpUbFmPUszMSx6DJJ4AHQCvQaKPYx8/vf8AmH9o1u0P/BcP/kTybX/goPFGqNqesa9HPeMgRpV08RFwOm7Y4BOOMnnAA6AVmf8ADOWl/wDQX/8AJZv/AI7XtlFHsY+f3v8AzD+0a3aH/guH/wAiefaB8PdY8LaWum6L4qjtLQOX2LpUbFmPUszMSx6DJJ4AHQCtT/hH/F//AEPH/lJi/wAa62ij2MfP73/mH9o1u0P/AAXD/wCRPJtf+Cg8Uao2p6xr0c94yBGlXTxEXA6btjgE44yecADoBWZ/wzlpf/QX/wDJZv8A47XtlFHsY+f3v/MP7Rrdof8AguH/AMieLSfs86bKkKNq0YESbF22RUkbi3zESZY5Y8nJxgdAANTw98Gm8K3hvNH1yCC6PSZ9MSV04I+VnYlchiDtxnvmvVaKPYx8/vf+Yf2jW7Q/8Fw/+RPPtS+Hmq6voy6Pf+I7SfT0QJHbtosAWIBSo2YPyEKSAVwR2xXKf8M5aX/0F/8AyWb/AOO17ZRR7GPn97/zD+0a3aH/AILh/wDInif/AAzlpf8A0F//ACWb/wCO0f8ADOWl/wDQX/8AJZv/AI7XtlFHsY+f3v8AzD+0a3aH/guH/wAicv4H8F2/grS5rSKaO4klcEziHY2xVAVCckkD5iMn+I+prqKKK0jFRVkctatOtN1JvV/L8gooopmYUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFZ2v3U1l4c1S7t32TwWkskbYB2sqEg4PHUVo1k+Kf8AkUNa/wCvCf8A9FtUz+Fm2GSdaCfdfmeG3Hxme0l8iXxVqskyqvmGDRLZkVyAWUFpFJ2nK5wM444waj/4Xf8A9TNrn/gitP8A49XjN1BNda5Nb28Uk08tyyRxxqWZ2LYAAHJJPGK9+0T9mzTUt92va5dzTsiHZYKsSxtj5hucMXGcYOF6dOeMoUouKbb+9/5nbXx1WFWUYxhZN/Yh3/wkmlfE+y1NLff8TZLGed9nk3ehIpQ7sDc65QDvndgA845xJ/ws3S/+isf+W63/AMRXjnxM8J2fgrxpPo9g989qsUciPeIAW3DnawwHXORuwOQy4+XJ7T4TfCbQfHnhW61TVLvUoZ4r17dVtZEVSoRGydyMc5c9/Sq9jHz+9/5mX9o1u0P/AAXD/wCROv8A+Fm6X/0Vj/y3W/8AiKz/ABl8Sde8OaPomraT4p/tmx1bz/Ll/s+O32+UyqeGUk8kjkDp3zXIfEb4K33guwuNasL6O+0eJ1D+Z8k8IZyq5HRwMoNwwSW+6AM17H8FbCzk+DmmxSWkDx3f2j7SjRgibMrod4/i+UBeewA6Uexj5/e/8w/tGt2h/wCC4f8AyJ43/wAL68X/APPz/wCQ4v8A43Xr/gmfxf4x8IWOv/8ACWfY/tXmfuP7Oik27ZGT73Gc7c9O9fMPiXTYdG8Vavpdu0jQWV7NbxtIQWKo5UE4AGcD0FfZfgT/AJJ54a/7BVr/AOilo9jHz+9/5h/aNbtD/wAFw/8AkSl/wj/i/wD6Hj/ykxf40f8ACP8Ai/8A6Hj/AMpMX+NdbRR7GPn97/zD+0a3aH/guH/yJyX/AAj/AIv/AOh4/wDKTF/jR/wj/i//AKHj/wApMX+NdbRR7GPn97/zD+0a3aH/AILh/wDInJf8I/4v/wCh4/8AKTF/jWdqY8U+H7zRpLjxR9ugu9TgtJIf7PiiyrE5+YZPQY/Gu+rkvHf/ADLX/Yetf/ZqipTUY3Tf3v8AzOrB4udauqc4xs7/AGILo+0TraKKK6DxzD8YazceH/C17qlokTzwbNqyglTudVOcEHoT3rO/4uH/ANSv/wCTFHxM/wCSe6p/2y/9GpXW1i05VGr9F+p6cJxo4SE1BNuUlqr6JQt+bOS/4uH/ANSv/wCTFH/Fw/8AqV//ACYrraKfsvNmX17/AKdw+45L/i4f/Ur/APkxWP4p8SeN/CPhy71y/j8PSWtrs3pAsxc7nVBgEgdWHevRap6rpVjrml3GmanbR3NncJslifow/mCDggjkEAjBFHsvNh9e/wCncPuPn/8A4aO1P/oG2n/fhv8A47XX+BfiP4r+IH2/+ybbRYfsPl+Z9rSVc792MbWb+4euO1eS/GjwXp3gzxbbRaNaT2+m3dosqh2Z0EgYq6qzcnACMQSSN/YECu7/AGZoJlt/Etw0Uggd7ZEkKnazKJSwB6EgMpI7bh60ey82H17/AKdw+49P/wCLh/8AUr/+TFH/ABcP/qV//Jiutoo9l5sPr3/TuH3HJf8AFw/+pX/8mKP+Lh/9Sv8A+TFdbRR7LzYfXv8Ap3D7jkv+Lh/9Sv8A+TFH/Fw/+pX/APJiutoo9l5sPr3/AE7h9xyX/Fw/+pX/APJij/i4f/Ur/wDkxXW0Uey82H17/p3D7jkv+Lh/9Sv/AOTFH/Fw/wDqV/8AyYrxXx98bfF1v4t1HTNHmg0y10+7ltgUhSV5djbdzFwR1UkBQMbsHdjNe5+AfEs3i/wRpmu3FvHbz3KOJI4ySu5HZCRnkAlc45xnGTjNHsvNh9e/6dw+4rf8XD/6lf8A8mKjnn8e2tvLcXEvhSGCJC8kkjTqqKBkkk8AAc5rsar39jb6np1zYXkfmWt1E8MybiNyMCGGRyMgnpR7LzYfXv8Ap3D7jxv/AIXb/wBTD4X/APALUP8A43WnoHxL1PxRqi6Zo+teFJ7xkLrE0N5EXA67d6gE45wOcAnoDXA/Ez4I2fhHw5Pr+japPJa2vlie2vAC53Pt3K6gDqyfKV/vHd0Fc/8AAr7Z/wALX037N5/k+VP9q8rO3y/LbG/H8O/Z143be+KPZebD69/07h9x9K+ENZ1HWbPUf7US1W6s7+Szb7KGCHYF5G4k9SfTtxXRVyXgT/mZf+w9df8AstdbTpNuCuTmEIwxMlFWWn5BXEeIvip4f8L6zNpeopdieLbllEe1sqG43OCeGHau3rwP4lfD+88aeJvFN9YXGLrSIoZUtBCXNzuhUlVI5DYQ4GDkkDjrRUcrxSe7/RjwcKTVSdSN+WN7Xt9qK/U6/wD4Xr4P/wCnv/yD/wDHK6D/AITv/qVPFH/gu/8Asq+La+/6XJP+b8B+3wv/AD5/8mZyX/Cd/wDUqeKP/Bd/9lR/wnf/AFKnij/wXf8A2VcR8Q/jrD4Z1S60TQrCO81C2fZLczuDAjfKSAEOXIyykErtYd+RXeeAPGlv488LprEEH2aQSvDPb7y/lOpyBuKruypVuB/FjqDRyT/m/APb4X/nz/5MyL/hO/8AqVPFH/gu/wDsqP8AhO/+pU8Uf+C7/wCyrraKOSf834B7fC/8+f8AyZnJf8J3/wBSp4o/8F3/ANlR/wAJ3/1Knij/AMF3/wBlXW0Uck/5vwD2+F/58/8AkzPNp/jd4VtbiW3uIr+GeJykkcixKyMDgggyZBB4xVzTPixo+t+b/ZOla1f+TjzPslukuzOcZ2ucZwevoa8J+K3wpbwC8GoafcSXOi3DiFWnZfNil2k7WwAGBCsQQOxBAwC2p+zj/wAlD1D/ALBUn/o2Kjkn/N+Ae3wv/Pn/AMmZ7l/wnf8A1Knij/wXf/ZUf8J3/wBSp4o/8F3/ANlXW0Uck/5vwD2+F/58/wDkzOS/4Tv/AKlTxR/4Lv8A7Kj/AITv/qVPFH/gu/8Asq62ijkn/N+Ae3wv/Pn/AMmZyX/Cd/8AUqeKP/Bd/wDZUf8ACd/9Sp4o/wDBd/8AZV1tFHJP+b8A9vhf+fP/AJMzkv8AhO/+pU8Uf+C7/wCyo/4Tv/qVPFH/AILv/sq62ijkn/N+Ae3wv/Pn/wAmZyX/AAnf/UqeKP8AwXf/AGVH/Cd/9Sp4o/8ABd/9lXW0Uck/5vwD2+F/58/+TM5L/hO/+pU8Uf8Agu/+yo/4Tv8A6lTxR/4Lv/sqzPF3xi8LeDdYbSb0X11fR486K0gB8rKqy5LlQchgflJ6HOK6zQPEekeKdLXUtFvo7u0LlN6gqVYdQysAVPQ4IHBB6EUck/5vwD2+F/58/wDkzMb/AITv/qVPFH/gu/8AsqP+E7/6lTxR/wCC7/7Kutoo5J/zfgHt8L/z5/8AJmcl/wAJ3/1Knij/AMF3/wBlR/wnf/UqeKP/AAXf/ZV1tFHJP+b8A9vhf+fP/kzOS/4Tv/qVPFH/AILv/sqP+E7/AOpU8Uf+C7/7Kutoo5J/zfgHt8L/AM+f/Jmcl/wnf/UqeKP/AAXf/ZUf8J3/ANSp4o/8F3/2VdbRRyT/AJvwD2+F/wCfP/kzMnw/4gt/EdnPcW9vdW/kTtbyR3SBHV1AJBAJx1x+da1cl4E/5mX/ALD11/7LXW06UnKCbIx1KFLEShDb/gBVG91rStOmEN9qdnaysu4JPOqMR0zgnpwfyq9XFXun2Wo/FgQ31pBdRLoe4JPGHUHz8ZwR15P50VJOKVgwlGFWUvaN2Sb030N//hKfD3/Qe0v/AMDI/wDGj/hKfD3/AEHtL/8AAyP/ABo/4Rbw9/0AdL/8A4/8KP8AhFvD3/QB0v8A8A4/8KX73yL/ANh/v/gH/CU+Hv8AoPaX/wCBkf8AjR/wlPh7/oPaX/4GR/40f8It4e/6AOl/+Acf+Fed/Gq1g8O/D6S60PQ9PilmuEt57iOxQtBEwbLAgfISwVd3bdxg4IP3vkH+w/3/AMD0T/hKfD3/AEHtL/8AAyP/ABo/4Snw9/0HtL/8DI/8a+K9K8RajpOqW9+ki3Jhfd5F2vnRSDoVZG4II/EdQQQDX2RpOj+GNZ0ax1S38P6esF7bx3EayWUQYK6hgDgEZwfU0fvfIP8AYf7/AOBc/wCEp8Pf9B7S/wDwMj/xo/4Snw9/0HtL/wDAyP8Axo/4Rbw9/wBAHS//AADj/wAKP+EW8Pf9AHS//AOP/Cj975B/sP8Af/AP+Ep8Pf8AQe0v/wADI/8AGj/hKfD3/Qe0v/wMj/xo/wCEW8Pf9AHS/wDwDj/wo/4Rbw9/0AdL/wDAOP8Awo/e+Qf7D/f/AAD/AISnw9/0HtL/APAyP/Gj/hKfD3/Qe0v/AMDI/wDGj/hFvD3/AEAdL/8AAOP/AArl/iHoGjWXgXUri00iwgnTytskVsiMuZUBwQM9CRUzlUjFy00NsNRwVetCinJczS6dXY76iiitzyworJ8U/wDIoa1/14T/APotq5fQPh54WvfDml3dxpe+ee0ikkb7RKNzMgJOA2OprOU5c3LFfj/wGdtHD0XRdatNrW2kU+l+sonfUVyX/Cs/CH/QI/8AJmX/AOLrj/ih4I0zRPh/qOo+HtHxfQbHMgnldooww3sFO4Ngdc4AXc2flwVer2X3/wDAH7PA/wDPyf8A4Av/AJYeu0V8J/8ACQap/wA/X/kNf8K9o+B3huLxdp2sX/iG0+12scscNq/mGPDgMZBhCD0aPr68d6L1ey+//gB7PA/8/J/+AL/5YfQlFcl/wrPwh/0CP/JmX/4uj/hWfhD/AKBH/kzL/wDF0Xq9l9//AAA9ngf+fk//AABf/LDraK5L/hWfhD/oEf8AkzL/APF0f8Kz8If9Aj/yZl/+LovV7L7/APgB7PA/8/J/+AL/AOWHW0VyX/Cs/CH/AECP/JmX/wCLrD8ReD9B8P3nh270ux+zzvrVtGzec75UknGGYjqBSlOpFXaX3/8AANaOFwlaapwqSu+8Fb/0tnpNFFFbHmBRXJfEz/knuqf9sv8A0alH/Cs/CH/QI/8AJmX/AOLrKU5c3LFfj6+T7HdTw9D2CrVptXbSSinsovrKP8x1tFcl/wAKz8If9Aj/AMmZf/i6P+FZ+EP+gR/5My//ABdF6vZff/wA9ngf+fk//AF/8sOtorkv+FZ+EP8AoEf+TMv/AMXR/wAKz8If9Aj/AMmZf/i6L1ey+/8A4AezwP8Az8n/AOAL/wCWHW0VyX/Cs/CH/QI/8mZf/i6P+FZ+EP8AoEf+TMv/AMXRer2X3/8AAD2eB/5+T/8AAF/8sOtorkv+FZ+EP+gR/wCTMv8A8XR/wrPwh/0CP/JmX/4ui9Xsvv8A+AHs8D/z8n/4Av8A5YdbRXJf8Kz8If8AQI/8mZf/AIuj/hWfhD/oEf8AkzL/APF0Xq9l9/8AwA9ngf8An5P/AMAX/wAsOtorkv8AhWfhD/oEf+TMv/xdH/Cs/CH/AECP/JmX/wCLovV7L7/+AHs8D/z8n/4Av/lh1tFeRfEz4fwaf4LnuvCOi79SSWPcEaaaUxk4PlpkgtkrnIwF3HrivnD/AISDVP8An6/8hr/hRer2X3/8APZ4H/n5P/wBf/LD7sorw/4MaBo/jHwVNe61p0c93BeyW/nLLIhdQqOCwVgM/ORwBwB3yT6H/wAKz8If9Aj/AMmZf/i6L1ey+/8A4AezwP8Az8n/AOAL/wCWHW0VyX/Cs/CH/QI/8mZf/i6P+FZ+EP8AoEf+TMv/AMXRer2X3/8AAD2eB/5+T/8AAF/8sOtorkv+FZ+EP+gR/wCTMv8A8XR/wrPwh/0CP/JmX/4ui9Xsvv8A+AHs8D/z8n/4Av8A5YdbRXJf8Kz8If8AQI/8mZf/AIuj/hWfhD/oEf8AkzL/APF0Xq9l9/8AwA9ngf8An5P/AMAX/wAsOtorkv8AhWfhD/oEf+TMv/xdH/Cs/CH/AECP/JmX/wCLovV7L7/+AHs8D/z8n/4Av/lh1tFcl/wrPwh/0CP/ACZl/wDi6+YPG3iNf+Et1C30Sz/sywtZWt44WDl22MQXfzfmVj/dwNvAxkEkvV7L7/8AgB7PA/8APyf/AIAv/lh9mUV4P8Kbrwb4+SfT9Q0CO21q3QzMsE8/lSxbgNy5clSCyggnuCCckL6X/wAKz8If9Aj/AMmZf/i6L1ey+/8A4AezwP8Az8n/AOAL/wCWHW0V5l448IeHPDHgvVNZsPDf226tIt6Qm4nI5IBZsNnaoJY9OFPI6j54vvHUt3ZyQQ6HpVlI2MTwG4Lpgg8B5WXnpyD19eaL1ey+/wD4AezwP/Pyf/gC/wDlh9qUV8n/AAp1KPxB4z0/w9rGnWl5BdvKzXDNKkqhYmYKux1XGU7qTyeemPfvh1aw2Vnr9pbpsgg1q4jjXJO1VCADJ56ChTnzJSW/n/wByw2HdGVSjNvlto4pb+kn+R2VFFFanAFFebeHfB+g+ILzxHd6pY/aJ01q5jVvOdMKCDjCsB1Jrc/4Vn4Q/wCgR/5My/8AxdYxnUkrpL7/APgHp1sLhKM/ZzqSuu0Fb/0tHW0VyX/Cs/CH/QI/8mZf/i6+dPifeXug+MbrSbXRY9Ft7dyYCHeVrmI4CyFnJBB2kgKBjJU5K071ey+//gGXs8D/AM/J/wDgC/8Alh9cUV8LweJdShuIpXeOdEcM0UiAK4B+6duDg9OCD6EV9d/8Kz8If9Aj/wAmZf8A4ui9Xsvv/wCAHs8D/wA/J/8AgC/+WHW0VyX/AArPwh/0CP8AyZl/+Lo/4Vn4Q/6BH/kzL/8AF0Xq9l9//AD2eB/5+T/8AX/yw62iuS/4Vn4Q/wCgR/5My/8AxdH/AArPwh/0CP8AyZl/+LovV7L7/wDgB7PA/wDPyf8A4Av/AJYdbRXJf8Kz8If9Aj/yZl/+Lo/4Vn4Q/wCgR/5My/8AxdF6vZff/wAAPZ4H/n5P/wAAX/yw62iuS/4Vn4Q/6BH/AJMy/wDxdRfDq1hsrPX7S3TZBBrVxHGuSdqqEAGTz0FCnPmSkt/P/gDlhsO6MqlGbfLbRxS39JP8jsqKKK1OAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigArJ8U/wDIoa1/14T/APotq1qyfFP/ACKGtf8AXhP/AOi2qZ/CzfC/x4eq/M8n+HPwc+w+J7bxdqV7Bc2pjS8srdY+fMkTdmQMCBsLfLtJJIDZXGD7dWT4W/5FDRf+vCD/ANFrWtRD4UGK/jz9X+Z8gfG3/kr2u/8Abv8A+k8dev8A7OP/ACTzUP8AsKyf+ioq8E+IM81z8RvEjzyySuNTuEDOxYhVkKqOewUAAdgAK+l/gdPDN8JNISKWN3hedJVVgSjec7Yb0O1lOD2IPeqMDtNc0az8Q6He6RfpvtbuJonwASuejLkEBgcEHHBANR+HNAsfC3h+z0XTVkFpaoVTzG3MxJLMxPqWJPGBzwAOK1KKAPhjxZY2+meMtcsLOPy7W11C4hhTcTtRZGCjJ5OAB1r631L4neEdCitF1jXbGK6niWQxWjvdBcqrdUXO0hgVZgu4HIFfIGu/2j/wkOp/2v8A8hP7XL9s+7/rt53/AHfl+9npx6V9F+CfgNoFt4fgl8V2Ml3q0yBpovtbCOA5bCr5e3J2lQ2SwyODjqAdpo3xT8E6/qKWGna/A90+AiTRyQ7ySAFUyKoZiSMKOT6V0mq6rY6Hpdxqep3MdtZ26b5ZX6KP5kk4AA5JIAyTXyR8XPBlj4I8amw0ySQ2dzbrdxRPyYQzMuzdnLAFDgnnBAOSMn6L8DzW/jz4R6WNbtftUN1afZ7lJ5DIZTGxjLluDuJTfnqCeuRmgCv/AMLt+Hn/AEMP/klcf/G66jQfFWheJ/tn9i6nBe/Y5fJn8on5W7HnqpwcMMqcHBODXyB8QvCf/CFeNL7Ro3nktU2yW00ybTJGwBHs2DlSw4JU8DoPd/2drHS4/A13f2cc4v5rsw3ryt8rFBlAgH8IWQdedxbsBQB3niXx94X8IXEFvrurR2s86F44xG8jbQcZIRSQM5AJxnBx0NcFr3xU8KeI9Z0LT9LvZJUttRt76a8kjMUEaKxVgS+CCNwOcYx37Uz9ofRNIm8HW+tT+XDqlvcJBbyCMlp1bcTESOAANzgtnG0gY3nPmfw18FzSppniW/SNtPu9Wh05LaaEkXClsuxzwU+Xb3ydwONvOVb4Gd2W/wC9R+f5M+sKKKK1OE5L4mf8k91T/tl/6NStGfxp4VtbiW3uPEujQzxOUkjkv4lZGBwQQWyCDxis74mf8k91T/tl/wCjUr59+IXwavPBVnfaxHq1jJpCSqlsk0hW5k3EYXbt2sw5JweQpbA6DJfxX6L9Tuqf7jT/AMc/ypn1PY39nqdnHeWF3Bd2smdk0EgkRsEg4YcHBBH4Vz/iH4i+EfCt4LPWNbgguj1hRXldOAfmVASuQwI3Yz2zXyJ4Si8R3XiCKy8LXF3Dql0jIn2W58hnUDewLblGMJnBPYV7Hpn7Ne7R5f7W8QeXqb48v7JDvhiwxzndhpMrjpswc/erU4T2jw94q0LxXZm60PU4L2NfvhCQ8eSQNyHDLnacZAzjI4rYr4s1zQ/Efwr8Y2yPdRwalAi3Ntc2km5WU5GRkA4yGUqw5weCDz9V/D/xjb+OPCVtq0Xy3C4hvIwhURzhQXC5JyvII5PBGecgAHk/7TX/ADK3/b3/AO0a6D9nH/knmof9hWT/ANFRVz/7TX/Mrf8Ab3/7RroP2cf+Seah/wBhWT/0VFQB7BWHP408K2txLb3HiXRoZ4nKSRyX8SsjA4IILZBB4xXhHx/8c31z4gfwhayyQ6faJG92gGPPlYBxk55RVKEDj5sk5wpFz/hmi8/s7f8A8JPB9u8rPk/Yz5fmY+7v352543bc4529qAPf7G/s9Ts47ywu4Lu1kzsmgkEiNgkHDDg4II/CrFfHHh3xh4g+Fni26sorn7Rb2d3LBeWIlbyJyrbHK5Hyt8gw+M8DIIyp+x6AI54Ibq3lt7iKOaCVCkkcihldSMEEHggjjFU9T13R9E8r+1tVsbDzs+X9ruEi34xnG4jOMjp6ivBPi/8AF/Uv7Zu/Dfhu8ktLa1cR3V3ErRzmZGbeiPnIQHaCQASVPJU819A/Zy1fUNLW41rV49Ju2cj7KsAuCqjoWZZAMnk4GeMc5yAAfSdFfIniKx8R/BXxiLPSPEMmZ7dLhZYk2rKp3piSJtykgh8Z3YyCMHger61YeJvjP8K9CvNMvrTTDcO7X1m7MIpyjlQ24AsAGQsEII+YZJKAkA2PEfwN8K+JfEF5rNxcalbT3bh5I7R4kj3YAJAMZOSRuJzyST3r0DStKsdD0u30zTLaO2s7dNkUSdFH8ySckk8kkk5Jr4Uv7G40zUbmwvI/LurWV4Zk3A7XUkMMjg4IPSvvegAorwP4s/GfU9I8QPoHhaeO3eyfF3e7Y5d745jUEMAFzhiRncMcbTuwLT4O/ErxElhr1/rMcN+EV4f7RvZzdW4DFl52sUIJ3AZyCecHIoA9b+Nv/JIdd/7d/wD0ojrwD4Jf8le0L/t4/wDSeStD4lR/EXwvo9j4e8T6/wDb9Mut0sZim8zzGRslXdlEjYLKcNleVx93jQ/Zx/5KHqH/AGCpP/RsVAHuXgT/AJmX/sPXX/stdbXJeBP+Zl/7D11/7LXW1lR+BHdmX+9S+X5IK5Lw/wD8lC8Y/wDbl/6KNdbXJeH/APkoXjH/ALcv/RRoqfFH1/RhhP4Nf/Av/S4HxrYWNxqeo21hZx+ZdXUqQwpuA3OxAUZPAySOtfe9fDnguCG68deHre4ijmgl1O2SSORQyuplUEEHggjjFfWfxKtfF154SaLwZceVqRlAkCuiO8JVlYIzcK2SrZypG04OcA6nCcP4y+ANv4h8US6vpmsfYI72Uy3cMsJl2uwYs6HcM5bb8pxjLEHAC16h4Y8MaX4R0OHSNIg8q3j5Zm5eVz1dz3Y4H5AAAAAfIH/CdeONM1HEviTXI7q1lw0Nxdyna6nlXRzg4IwVYexFfa9ABRXg/wAZvi3cabeDw54V1PyriPeuo3MKglCRgRI/ZhlixAypCgMCGFcZpXwO8ZeKNLt9ee/01TqKfav9LuZGlcP8wdiqMMsDu6k884ORQB9V1GJ4WuHt1ljM6IrvGGG5VYkKSOoBKsAe+0+lfGmo6n41+H+o33hY67fWn2b900Vtdt5YViJA0eD8u7hsjDYYg4ywrvPhF4C8bz+JbLxi93Jp1nM/nTTXDl5b+JySw2ckhiB8z4+8rruIFAHZ/tHf8k80/wD7Csf/AKKlrn/2Zf8Amaf+3T/2tXQftHf8k80//sKx/wDoqWvMPhr4+fwj4evtO8PaFPqXirUbtSv7tpIzAiZA2I25mH7zgAcNkk7cUAfV9FfLmq6B8cIEuNRupvEBBfe6WmpBzlm/hiicnGT0VcAegFej/BX4nX3jJLzR9dkjl1S1Tz451TaZ4i2G3BQFBUlRxjIYccEkA9cr4k8aTzWvxL8Q3FvLJDPFrFy8ckbFWRhMxBBHIIPOa+n/AIjeLPE2gWVs3hDRI9Zn+0GG82o05tjsDKrRxkNlgwOegAGR8618gTzzXVxLcXEsk08rl5JJGLM7E5JJPJJPOaAPu/SY76HRrGLVJo59QS3jW6ljGFeUKN7DgcFsnoPoKuV86P8AF74raXb2kV/4TjDyOltHLdaXcI1xKRwBhlBdsE4UDvgVp+PfjF448IeKLuxPh2xtrASslnNdwyv9oRQPnDq6q2cg4A+XcFPIoA94or5k0X9orxHY6XNb6pp9pqd3sIguifJIY7uZFUYcDKjC7OF6knIj/wCEy+Of/Prrn/giX/4zQB9P0V5H8MfjUvjLVI9C1ixjtNUkRmhlt9xinK7mK7TkoQozySDhuRwD0nxV8b3ngLwlHqdhawXF1NdpbIJydiZVmLEAgnhCMZHXPbBAPEPiT8KfFMXj67uNN0+fVLXV7uS4hmtoyRE0j5KSdkwW+8xCkc5GGC+3/CrwReeAvCUmmX91BcXU129y5gB2JlVUKCQCeEBzgdcdsnxyD9pDxUtxE1xpWjSQBwZEjjlRmXPIDFyAcd8HHoa978ca9eeGPBeqazYWX226tIt6QkEjkgFmxztUEsenCnkdQAdBRXzB/wANHeMP+gbof/fib/47Whr/AO0fql1Zwx6DpMFhM8Ti4kuW88xuSQpjxtHAwcspyTjGBlgD6Por5gsfjN8QvCuuRjxXbT3ULxEmxvLRbNyDkK6sIwRyCOQQfmGM8j6P0TW9O8R6PBq2k3H2ixn3eXLsZN21ip4YAjkEcigDQorl/iB4xt/A/hK51aX5rhsw2cZQsJJypKBsEYXgk8jgHHOAfnSb4p/EvxhqgXSp7sPC63K2ekWpITbgZIAZ2Qk8q5Kknp0FAH1nRXyQPi58SvD15DY3+ozpJabA9pf2SB2UAECQlRIcjGTncc5znmuT8V+LdX8Z6ydU1maOScJ5caxxhFjj3MwQY5IBY8kk+pNAH1t4E/5mX/sPXX/stdbXm3wRnmuvAslxcSyTTy3ZeSSRizOxijJJJ5JJ5zXpNZUfgR3Zl/vUvl+SCuS/5q9/3Af/AGvXW1yX/NXv+4D/AO16KvT1DA/8vP8AA/0Otorw/wAaftCW+lajdab4ZsINQaLCjUJpSYS4PzBUXBdccBtwyeRkAFvND8ZviStulw2tyCB3ZEkNhBtZlALAHy8EgMpI7bh61qcJ9d1xfxM8cab4I8NCa/sY9Se9c26WDyKomUj5y2QfkC8H5Tyyg9c1yfw3+ONv4q1GLRtftoNP1KXiCaJiIZ3JOEw2SjY2gZJ3HPIJCn0TxL4P0DxhbwQa9psd4kDl4iXZGQkYOGUg4PGRnBwPQUAfLnhf4f6p8UPEOpX+lWsGk6Qbtnd2H7uAM+fKjAA3sqnoNowBkrkZ+t7Cxt9M062sLOPy7W1iSGFNxO1FACjJ5OAB1qSCCG1t4re3ijhgiQJHHGoVUUDAAA4AA4xUlABRXg/if9o+3glmtvDGk/advCXl6xVCQ3JEQ+YqVHBLKeeRxg5GjftI6wmop/bmjWMticBvsIeORORlhvZg2Bn5flycfMKAPo+isfw34p0bxdp0l/od59rtY5TCz+U8eHABIw4B6MPzrxuf9pmFbiVbfwpJJAHIjeS/CMy54JURkA47ZOPU0Ae+VyXxM/5J7qn/AGy/9GpXEWf7RXhybw/dXd1p93b6pCmY7DO9Z2JIAWUDAAGCxYDGeA2OePvPjjeeLtMvNAvNDgh+2S5hnhnP7qNWDqrKQd7fKQWBUc/dGOcq/wDCl6M7sr/36j/jj+aPpSiiitThMnxT/wAihrX/AF4T/wDotq4vxP41m8B/CDRtUtII5ryWC0t7dZULRhjGGJfDKcbUfGD1x2zXaeKf+RQ1r/rwn/8ARbV83/E34iaXr3gvRfCmnwzvNp5he5uXG1N6Q7Cijq3LsCTj7vG4HNZf8vfkd3/MD/2/+h0fwj+LnijXfG8Gha7cR6hBfI4jkMSRNAyIz5GxQGBCkEH2IIwQfoOvjj4VeN7PwF4tk1O/tZ7i1mtHtnEBG9MsrBgCQDygGMjrntg/Tfgf4h6L4/t7yXSUu4ns3VZorqMKwDA7WG0kEHaw65+U5A4zqcJ4Z+0d/wAlD0//ALBUf/o2Wu//AGcf+Seah/2FZP8A0VFXAftHf8lD0/8A7BUf/o2WrHwt+KWheAfh5fW14s91qcmoSSw2cKEZUxIFZnPyqu5CDjLDOdpoA+l6K8L039pbTZbhl1Tw3d20GwlXtblZ2LZHBVggAxnnPYcc8e0aVqtjrml2+p6Zcx3NncJvilTow/mCDkEHkEEHBFAFyiivI/Ef7QfhnSLi8tNNtLvVbiBwiSRsqW8pyN2JMk4HPIUgkccHNAHrlcl47/5lr/sPWv8A7NXnmmftKaPL5v8Aa3h++tcY8v7JMlxu65zu2Y7dM5yemOe/8Zzw3Vv4VuLeWOaCXW7R45I2DK6kMQQRwQRzmsq3wM7st/3qPz/JnY0UUVqcJyXxM/5J7qn/AGy/9GpXW1yXxM/5J7qn/bL/ANGpWzr/AIj0jwtpbalrV9HaWgcJvYFizHoFVQSx6nAB4BPQGsl/Ffov1O6p/uNP/HP8qZqUV8//APDTX/Uo/wDlS/8AtVdh4F+N2heLrwadfQ/2PqUsojtoZJTIk+RxiTaAGyMbTjOVwSTganCeoUVT1bUodG0a+1S4WRoLK3kuJFjALFUUsQMkDOB6ivK/+GjvB/8A0Ddc/wC/EP8A8doA9gorh/hx8SrP4iWd40Onz2V1ZbPtEbuHT5y+3awwTwnOVGM4561x91+0j4bTyPsejarLmVRN5wjj2R/xMuGbcw4wp2g/3hQB7RRXifgv4/w63rK6XrOkSQT3t6kFi1nh1VXYKol3MDkEjLKOc/dGOe4+IPxJ034eW9o17Z3d1PepKbZIdoUsgXh2JyoJccgN349QDtKK8f0P9obw/qktlaXWj6rb31zKsRSBFuEUs2FwQQ7cEHATPYA9/RPEvjDQPB9vBPr2pR2aTuUiBRnZyBk4VQTgcZOMDI9RQBuUV4nqX7SegxW6tpeh6lcz7wGS6ZIFC4PIZS5JzjjHc88c9p4O+LHhbxtefYbCee2vzuKWl5GEeRVAJKkEqep4zu+VjjAzQB3FfIHxt/5K9rv/AG7/APpPHX1/XzB+0d/yUPT/APsFR/8Ao2WgD1f4FX1vd/CjTYYJN8lpLPDONpGxzI0gHPX5XU8evrmvSK8/+CX/ACSHQv8At4/9KJKx7/8AaD8J6fqNzZS6drhkt5XiY/ZUTJUkH5XcMOnRgCO4BoA9YorzPw58cvCviXxBZ6Nb2+pW0925SOS7SJI92CQCRITkkbQMckgd69InnhtbeW4uJY4YIkLySSMFVFAySSeAAOc0ASUV5XqX7QXgixuFitzqWoIUDGW1tgqg5PynzGQ54z0xyOeuNjwp8X/CHi24Fpb3kljeM+2O2v1WJpeVA2kEqSS2AudxweMc0Ad5RWXr/iPSPC2ltqWtX0dpaBwm9gWLMegVVBLHqcAHgE9Aa4PTPj94Gv8AzftNxfabsxt+12pbzM5zjyi/THfHUYzzgA9QorzfxD8cfBWgXgtUup9Uk/jOmqsqJwCPnLKrZz/CTjBBwap6b+0F4Ivrhorg6lp6BCwlurYMpOR8o8tnOec9McHnpkA9Ur5g+On/AAg39sP/AGF/yMf2tv7S+z58n7ozuz8vmbsfc7+Zv+bFfT9eT+FPgL4f8Pa5cahfz/2zDyLW1u7ddkQOeXGSJGxgA4AHJxnG0A4T4FfD7Wm8T2viy8gkstPtEYwedGQ10ZIyoKA4+Ta+7f0PAGeSv0nRWH4r8W6R4M0Y6prM0kcBfy41jjLtJJtZggxwCQp5JA9SKANyvF/2kbG3k8G6TftHm6h1DyY33H5UeNywx05Mafl7mrH/AA0d4P8A+gbrn/fiH/47XIfFj4leH/Hvw8gj0p54bq31WMva3aqkhTypfnADEFcnHB4OMgZGQDj/AIJf8le0L/t4/wDSeSvpLwJ/zMv/AGHrr/2WvnX4FWNxd/FfTZoI98dpFPNOdwGxDG0YPPX5nUcevpmvorwJ/wAzL/2Hrr/2Wsp/HH5ndh/91rf9u/mdbRRRWpwnJeBP+Zl/7D11/wCy11tefaJ4j0jwtpfifUtavo7S0HiC4TewLFmO3AVVBLHqcAHgE9Aa891X9paZkuI9H8Nxo+/EE93clht3dWjUDkr2D8E9TjnKj8CO7Mv96l8vyR9B1xfxT0Sx1b4fa1Nc6baXdxZ2U01vJMdrQkDcWRwCQflBwMBtoUkAk15/pv7S2my3DLqnhu7toNhKva3KzsWyOCrBABjPOew4549M8c/bNU+Gutf2B5F5Ndae/k4zIs0bL82zZncxQtsxwWK9q1OE+KK+/wCvgSCCa6uIre3ikmnlcJHHGpZnYnAAA5JJ4xX234s8aaF4K05bzWrvyvN3CCFFLyTMoyQqj8Bk4UEjJGRQB0FFeT337Q3gq0vJIIYtVvY1xieC3UI+QDwHdW46cgdPTmrmr/HfwRpaWjwXN3qYuEZ/9ChGYgG2/OJChUkg4HXAzwCpIB6ZRXkcH7RXg2a4iie01mBHcK0slvGVQE/eO2QnA68An0Br0z+29O/4R7+3/tH/ABLPsn23z9jf6nZv3bcbvu84xn2oA0KK8nvv2hvBVpeSQQxarexrjE8FuoR8gHgO6tx05A6enNdJ4B+Jmi+P7eRbMSWuoQIrT2UxG4AgZZCPvoGOM8HpkDIyAdpXHfD2CG1t/ENvbxRwwRa3cpHHGoVUUBAAAOAAOMV2Ncl4E/5mX/sPXX/stZT+OPzO7D/7rW/7d/M62iiitThCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKyfFP/ACKGtf8AXhP/AOi2rWrJ8U/8ihrX/XhP/wCi2qZ/CzfC/wAeHqvzDwt/yKGi/wDXhB/6LWtasnwt/wAihov/AF4Qf+i1rJ1/4l+D/C+qNpmsa1HBeKgdolhklKA9N2xSAcc4POCD0Ioh8KDFfx5+r/M+RPFl9b6n4y1y/s5PMtbrULiaF9pG5GkYqcHkZBHWvpP9n2a+l+GQS7EggivZUs90e0GLCsdpx8w8xpOeecjtgfLE8811cS3FxLJNPK5eSSRizOxOSSTySTzmvov9nvXdHsfBtzp15qtjb30+qt5NtNcIkkm6OJV2qTk5IIGOpqjA9woorD8UeLtF8HaW9/rF7HCAjNFAGBlnIwNsaZyxyy+wzkkDmgD4cr7/AK+AK+/6APlT9oLUob74mm3iWQPYWUVvKWAwWJaXK89NsijnHIP1PtfwS/5JDoX/AG8f+lElfPnxnnhufi3rzwSxyoHiQsjBgGWFFYcdwwII7EEV9B/BL/kkOhf9vH/pRJQBj/H7ww+teBo9UtoPMutJl81iNxYQMMSYUcHBCMSeioxyOc+efs66+un+Mb7RZWjVNUtw0eVYs0sWWCgjgDY0pOf7o57H6br4Qv7W88OeIbmz+0bL7Tbt4vOt3I2yRuRuRuCORkHg0AekfGXxDceNviVB4e0xfOjsJfsFumQvmXLsBJywGPmCpySPkyDg17l4ssbfTNO8IWFnH5dra6zZwwpuJ2ooYKMnk4AHWvLf2fPAry3jeM70bYYfMgsUKsCzkYeUHgFQCyDrkluhUZ9a8d/8y1/2HrX/ANmrKt8DO7Lf96j8/wAmdbRRRWpwnJfEz/knuqf9sv8A0alZnxt/5JDrv/bv/wClEdafxM/5J7qn/bL/ANGpWZ8bf+SQ67/27/8ApRHWS/iv0X6ndU/3Gn/jn+VM8g/Zx/5KHqH/AGCpP/RsVfT9fJnwG1Kax+KlnbxLGUv7ea3lLA5ChDLleeu6NRzngn6j6zrU4T5//aa/5lb/ALe//aNH7Mv/ADNP/bp/7WrA/aE8T2+seLbLSLOeCeHSonErR5JSd2+dCehwETp0JYE5GB6P+zzY3Fp8NZJp49kd3qEs0B3A70CpGTx0+ZGHPp6YoA5T9pmeFrjw1brLGZ0S5d4ww3KrGIKSOoBKsAe+0+ldf+z7Zw23wyE0V3HO91eyyyxrjMDAKmxuTztRX5xw447niP2ltShl1nQNLVZPPt7eW4diBtKyMqqBznOYmzx3HXt2/wCz7ps1j8MhcStGUv72W4iCk5CgLFhuOu6Njxngj6AAk8XeB/h/beL28Y+K9U8uaTE32O7uIxDN5UargR7d8mAFO0E5JAwQcHP1v9onwtY+fHpNnfapMm3y32iCGTOM/M3zjAJ6pyR6HNeEXOuXHj3x9Z3fie/8uG8u4oZpFcRpbQFwCE3ZCKoJPOe5OSST9f6PomheDNDe1023g07TYd88hZzgd2d3Y5OAOrHgADoBQB8Ya3rD+K/FE+qXiWOnyX0qmYwRMsMZIAZyo3NzyzYySSTgk19r67qf9ieHtT1byfO+w2ktz5W7bv2IW25wcZxjODXwxf31xqeo3N/eSeZdXUrzTPtA3OxJY4HAySelfbfiWFtd8C6vBpZjunv9MmS1Mci7ZS8RCYbOMHI5zjmgD448G6baax410TTr9oxaXN7FHKHLgOpYZTKDILfdB4wSMkDJH3HXwppN5N4a8VWN7cWknn6ZexyyW0mY23RuCUORlTlcdOPSvtvRNb07xHo8GraTcfaLGfd5cuxk3bWKnhgCOQRyKAPN/jd4A13xtZ6TPoccE0mn+dvgeUI8nmGMDaT8vG0k5I9sniuw+Hfhu88I+BNN0O/kgkurXzd7wMSh3Su4wSAejDtUnjPxzovgXS0vtYlkJlfZDbwANLMeM7QSBgA5JJAHA6kA2PB/iWHxh4TsNegt5LdLtGJhcglGVijDI6jcpweMjHA6UAfIHjSCa6+JfiG3t4pJp5dYuUjjjUszsZmAAA5JJ4xX2fq2pQ6No19qlwsjQWVvJcSLGAWKopYgZIGcD1FfEl/4huLjxlc+JbNfsl1JqD38IyJPKcyGReow2DjqOcdK+z/Fljcan4N1yws4/MurrT7iGFNwG52jYKMngZJHWgD5E+GOmzar8TfDtvA0aul6lwS5IG2I+aw4B52oQPfHTrX2nXwx4U1n/hHvFuk6uXnSO0u45ZfIOHaMMN6jkZyu4YJwc4PFfc9AHlf7QWmzX3wyNxE0YSwvYriUMTkqQ0WF467pFPOOAfofNP2cf+Sh6h/2CpP/AEbFXo/7Qes/2f8ADpdOR4PM1K7jiaNz85jT94WUZ7MsYJ5A3epFecfs4/8AJQ9Q/wCwVJ/6NioA9y8Cf8zL/wBh66/9lrra5LwJ/wAzL/2Hrr/2WutrKj8CO7Mv96l8vyQVyXh//koXjH/ty/8ARRrra4Jf7R/4SH4h/wBkf8hP7Jb/AGP7v+u+zts+98v3sdePWip8UfX9GGE/g1/8C/8AS4Hyz4LnhtfHXh64uJY4YItTtnkkkYKqKJVJJJ4AA5zX3HXw54LghuvHXh63uIo5oJdTtkkjkUMrqZVBBB4II4xX3HWpwnxB47/5KH4l/wCwrdf+jWr7fr4g8d/8lD8S/wDYVuv/AEa1fb9AHwxf2uo33jK5s9XuILfU59QeK8muHVI45mkIdnZflChiSSOAOlfc9fNn7RWi6LZeILHVLeeRdY1BCbq3BDKUQBVkPOUJxtAxhtp6FTu39K+KHxB0/wANW+nz/DjWbvUILfyheyxXGJGAwruhjJY9C3zjcc8jPAByn7R3/JQ9P/7BUf8A6Nlr3/wJ/wAk88Nf9gq1/wDRS14p4U+DfiPxd4gHiTx9JJHBO/nTW8kmLi44UqpC8RJg4xkMoXaFXgj6LoA8f/aO/wCSeaf/ANhWP/0VLWP+zRY3EeneIr9o8Ws0sEMb7h8zoHLDHXgSJ+fsauftJyXw8K6NFHDGdPa9LTyk/MsoQ+Wo56FTKTwfujkd4/2a9T83w9rmk+Tj7Ndpc+bu+95qbduMcY8nOc87u2OQD3CvkD4Jf8le0L/t4/8ASeSvrPVtSh0bRr7VLhZGgsreS4kWMAsVRSxAyQM4HqK+NPhxNbwfErw491a/aYzqEKBPMKYdmAR8j+6xVsd9uDwaAPtevgzSVVtZsVewk1BDcRhrKNmVrgbh+7BXkFumRzzxX3nXw54LghuvHXh63uIo5oJdTtkkjkUMrqZVBBB4II4xQB9x15/8bf8AkkOu/wDbv/6UR16BXmfx51KGx+Fd5byrIXv7iG3iKgYDBxLluem2Nhxnkj6gA8g/Z/0qx1P4jO99bRzmysnurffyElEkah8dCQGOM9DgjkAj6rr5/wD2Zf8Amaf+3T/2tX0BQB8QeBP+Sh+Gv+wra/8Ao1a+36+IPAn/ACUPw1/2FbX/ANGrX2/QB8QeO/8AkofiX/sK3X/o1q+36+IPHf8AyUPxL/2Fbr/0a1fb9AHyR8dbG4tPivqU08eyO7igmgO4HegjWMnjp8yMOfT0xXufwV8OQ6F8N9OuGsY4L/UU+03EgIZpVLMYiTk8eWVIXtk8Ak14Z8db64u/ivqUM8m+O0ighgG0DYhjWQjjr8zsefX0xX0f8OL631D4a+HJrWTzI10+GEnaRh41Ebjn0ZWHvjjigDg/2kIIW8C6ZcNFGZ01NUSQqNyq0UhYA9QCVUkd9o9Ky/2aL64k07xFYNJm1hlgmjTaPldw4Y568iNPy9zVz9pPUoYvCujaWyyefcXpuEYAbQsaFWB5znMq447Hp3y/2Zf+Zp/7dP8A2tQBmftJ6lNL4q0bS2WPyLeyNwjAHcWkcqwPOMYiXHHc9e3Z/s7aZp1v4Gu9StpvNvru7Md38rL5Xlj5I+ThuH37gP8Alpg/dqT45+AZvE2jRa9ZS2kM+kW88lz5qENNCF34DAE5UqcKePnPI7+afCX4tWfgPTr3SdWsJ57GaU3MctoAZFkIVSpDMAVIUHIOQQeuflAO3/aUt93h7Q7n7Dv8u7eP7Z5uPK3Jny9n8W/bnd28vH8VcR8AdN0vWPGWo2Wq6TY38f8AZ7SobuHzfLKyIOFPy87upBPAwQCc1/ir8R/+Fk6jpWnaHbXy2MX3baRMSTXDnaPlRmDYGAvfLP61638DfA194S8NXd9qsUlvf6o6ObdzzFEgOzcMZVyXckZOBtBwQRQB0/gT/mZf+w9df+y11tcl4E/5mX/sPXX/ALLXW1lR+BHdmX+9S+X5IK8L+OPiG40DVLlLVf3mpaMLAvkfIjysX4IOcqjL2xuyDkV7pXzn+0d/yG9P/wCvaP8A9Cloq9PUMD/y8/wP9DkPgr4XsfFPxBjh1KOOa0srd7x7eRNyzFSqqp5HG5w3OQduCMGvruvmD9nH/koeof8AYKk/9GxV9P1qcJ8WfEfwlN4M8a32nNDHFaSu1xYhJC4+zszbBk85GCpzzlT1GCfqv4deIbjxV8P9H1i8XF1NEUmOR87ozRs/AAG4qWwBxnHavnT48x2KfFS8a0mkkne3ha8VhxHLsACrwOPLEZ78seew9z+CX/JIdC/7eP8A0okoA9Ar54/aB8eXDaj/AMIdp8+21SJX1EbAfMclZI05XI2hVbKnnfg9K+h6+LPidNfT/E3xE+oiQTi9dF3x7D5SnbFxgceWEwe4wec5oA6j4QX3w+0N5tY8WXsY1SG4U2MT200ghCr/AKz5AVJJbgEZUxgjrXf/ABL8ZeA/G3w81dbDV9Nm1K0SNrZrqB0lUmRSyxbk3EsEwdvA43EDmsT4d/BTw34u8Cabrl/e6rHdXXm70gljCDbK6DAMZPRR3rp/+GcfB/8A0Etc/wC/8P8A8aoA8w+AOp/YPihBbeT5n9oWk1tu3Y8vAEu7GOf9VjHH3s9sHc/aA8H6L4ffR9R0bTbSxN28qXKQOEDFVj2bYs4AA3ZKrjJG7lhnu/Dnwx8D+CviBphtdcvjroilmtrO5uIj5ibSjHAjB6MxHIztYjIVsV/2jv8Aknmn/wDYVj/9FS0AeOfCnwDD4/8AEs9pfS3cGn2tuZZpbdBksSFRNxBCk5Lcg5CMAO46fVvgjrHg6WbWv7UsbzTLXb82HjmbdhPuYKjDN/f6DPtS/s3zzL461O3WWQQPpjO8YY7WZZYwpI6EgMwB7bj617d8TP8Aknuqf9sv/RqVlX/hS9Gd2V/79R/xx/NHW0UUVqcJk+Kf+RQ1r/rwn/8ARbV81/F/w5pGkaJ4P1LT7GO3u9Tsy946EgSsscODtzgH5mJIAySScmvpTxT/AMihrX/XhP8A+i2rwf43Q27eAPAUzXW26S12R2/lk+YhiiLPu6DaVQY77/Y1l/y9+R3f8wP/AG/+hlfA3wJoHjO41uXXrWS6SzSFYohM0a5cvljtIJI2ADnHJ4PGPovw94V0LwpZm10PTILKNvvlAS8mCSNznLNjccZJxnA4rxf9mX/maf8At0/9rV9AVqcJ8wftHf8AJQ9P/wCwVH/6Nlq58EPhnoviiyl8R6wZLkWl75MVkQPKcqitmTuwy6/LwPl53A4qn+0d/wAlD0//ALBUf/o2Wu//AGcf+Seah/2FZP8A0VFQBn/G34beH7fwhL4k0qzg0y60/wAtXitIVSOdHkC4KjADAvncOoyDngrkfs3eIbhdR1bw0y7rV4vt8ZyB5bgpG3bJ3Bk78bOnJrs/2gtNmvvhkbiJowlhexXEoYnJUhosLx13SKeccA/Q+afs4/8AJQ9Q/wCwVJ/6NioA6v8AaD8c32mJa+FNOlkgF7bme9kUYLxFiqxhs5AJV9wxyNozgsDkfB74W+FPE2iHWNYvI9TndCrabHKYzaHewDOUfcSwTK52jBPDcEcX8bf+Sva7/wBu/wD6Tx10ng/4Dr4s8J2Gup4pjhF2jN5SWLOEIYqV3M6kkEYPGM5wSMEgHR/G34Y6FY+F5fE2iWMGn3FtLH9qjhJSOSNiIxtjA2hgxTptBBcnJxXm/wAKtVvl8VaZo4uZP7PfUbe6NueV81XVQ49DtYg468ZzgY9A/wCGZf8Aqbv/ACm//bax9E+Hc3gf4jacs+v6NeuuowRC1t5z9qVTIrK7xEfICoHc8sOTnNZVvgZ3Zb/vUfn+TPpiiiitThOS+Jn/ACT3VP8Atl/6NSvmH4salqWpfE3XDqayRvb3DW8EbBgFhQ4jKhjwGXD8cEuSOtfT3xM/5J7qn/bL/wBGpXlnxi8ZeCY/EIEeiQa34lsPLhM1xJIbSEK7MY3VXAkYEkFcY+bBJ2layX8V+i/U7qn+40/8c/ypm54B+EHhLUvBGmXeveGLuHVHRxcpd3M8cm4OwyVBQKCACBjoRy3U+GeNtB/4QX4gX2ladezt9gljkt7jOyRcqsi8j+JdwG4YyRnA6D0DRviB8X/HOsJ/YTfZ7WaURGSGwT7JbkKC26R1Yjj5iCxPOAOQK8j1aG+ttZvoNUMh1CO4kS6Mkm9jKGIfLZO47s85Oa1OE+49D1mz8Q6HZavYPvtbuJZUyQSueqtgkBgcgjPBBFfKnxf+H8PgTxLCdP8AM/snUEaW2DsCYmU/PHnOSFypBPZgMkgk/Rfwrury8+F/h6W+t/ImW0ESpsK5jQlI2wf7yKrZ6HORwRWX8bdAbXvhlfNEsjT6c63yKrKoIQEOWz1AjZzgYOQOvQgGH+ztqenXHga7022h8q+tLsyXfzM3m+YPkk5GF4TZtB/5Z5P3q5D4/eD/AA14fi0rUNJtoNPvruV0ktYImVJkVV+cAHYm07RgAbvMz2Ncn8FfEc2hfEjTrdr6SCw1F/s1xGAWWVirCIEYPPmFQG7ZPIBNan7346fGH/lva6THF/seZDax/wDszO3+1tMn8QWgCx+ztb6dc+Obv7TY+bfW1obm0uvNZfJwfLddg4bcJep6beOvHo/7QNjo0nw/+36hHnUoZVh059z/ACu7KZBgccpG33vTjk186QtfeDfGpRb+S0vNLvWhe6tF8wqUYo5VW2hwQD8rYDA4OATXonxN8Wad8UfH3h3Q9Guv+JasqWy3hgYEyTugdgrEEqoCYBCnIbtg0AXPgB4IttauNU13VrG0vNPiT7HDDdQJKrSkq7MA2dpVQo6c+YcHgiuf+POpTX3xUvLeVYwlhbw28RUHJUoJctz13SMOMcAfU/T/AIc0Cx8LeH7PRdNWQWlqhVPMbczEkszE+pYk8YHPAA4rzP4pfFPw/wCHNYXTV0GDV9fsNskMt3Cvl2bsu5WDEbiwxG2FxkEfMCMUAanhr4aeENZ+FukW1xolosl7pkMkl5HGouBI6By4lILZ3HOORjjG3ivmzW9M1HwD45nslm232lXayQT7VOcEPHJtyw5G1tpzjOD3r0jTvE/xf+KH2r+xp/semtLHFLJa7LaOA8H5ZD+9OMbmCsxwcYwwB8n121vLHxDqdnqNx9ovoLuWK4m3l/MkVyGbceTkgnJ5NAH3fXzB+0d/yUPT/wDsFR/+jZa+h/CdjcaZ4N0OwvI/LurXT7eGZNwO11jUMMjg4IPSvlj42/8AJXtd/wC3f/0njoA9D+G/xZ8L+EPhhZ2OqahJPf27yFbK1tH8wK0pOC7HYx+YtncvBAwSOec8DfDDVPiRrUvi7WxBaaReXbXbKnzfayZj5kagPujXhxuJyOMBuSPJ5rC8t7O2vJrSeO1ut32eZ4yEl2nDbWPDYPBx0r6f+CfxEt/EuhweG7iHyNT0q0RF2AlJoE2oHB7MMqCD1zkdSFAOs0D4aeD/AAvqi6no+ixwXioUWVppJSgPXbvYgHHGRzgkdCa+dPjD8QLjxf4ouLC1ut2hafKUtkjI2SuBhpSQSHyd20/3SMAEtn6znnhtbeW4uJY4YIkLySSMFVFAySSeAAOc18CUAfS/w3+D3g46PFf6hPY+Ib9JcSPa3nnWkbqxIVQoXdlSm5Xz9MHnzz4y/DTTfA1xY3ujzyfY755FNtNKrNCwII2c7mTDY5B24GWJYV0l9+zReR2cjWHieCe6GNkc9mYkbkZywdiOM/wn0461JB+zNM1vE1x4rjjnKAyJHYF1VscgMZASM98DPoKAOj8BInxd+Dx0zxM88slrd/ZzeBlMzFNrq4LKcNtfYTySNxJyxr508SaVDofiXUtJguZLpLK4e3MzxCIuyHax2hmwNwOOeRg8dB9X/DP4Zw/Dq31FV1STUJ754y7mERKqoG2gLljnLtk59OBjn5c8d/8AJQ/Ev/YVuv8A0a1AHs/wx+CWhX3hey1vxNDPdXF9F5sdp5xjjjjYgo2UO4sVGfvAAPgrkZrhPjL4C0XwNrNiujXchS+SSV7KVw7W4DDaQeuw5IG7J/dn5j2+n/DUljN4V0iXS4ZINPeyha1ikOWSIoNink8hcDqfqa+dP2jv+Sh6f/2Co/8A0bLQB6f8A9ZvNX+GqxXj+Z/Z929nC5JLGMKjqCST03lRjACqoxxXqFfP/wCzL/zNP/bp/wC1q+gKAM/XdT/sTw9qereT532G0lufK3bd+xC23ODjOMZwa+ONe8T+JfiR4hs47+f7TdSy+RZWqbY44zI/CqOgySBuYkkAZJxX0v8AG3/kkOu/9u//AKUR18ueC54bXx14euLiWOGCLU7Z5JJGCqiiVSSSeAAOc0AfQ/hL4DeG7Hw/FF4nsY9Q1YuzTSxXcyxgZ+VU27ONoBORnJPOMY84+Nnw50jwW+mX+g213FaXjyJMjOZIoWVU2BWIyC3znDMc4OMAV9R14/8AtHf8k80//sKx/wDoqWgDzT9n3UobH4mi3lWQvf2UtvEVAwGBWXLc9NsbDjPJH1Hv3gT/AJmX/sPXX/steG/s4/8AJQ9Q/wCwVJ/6Nir3LwJ/zMv/AGHrr/2Wsp/HH5ndh/8Ada3/AG7+Z1tFFFanCfH/AMVdVvm8Vano5uZP7PTUbi6FuOF81nZS59TtUAZ6c4xk5Z8ML/wHZapP/wAJvYyTgvG9pMVd4oSu4t5iKcsGOwY2sPXAzXo/iP4baX4ts/EetzavBpF9ZazdRfart8QOmRsRySNnztwwz94jDcY8Q1vw3rXhy48jWdLu7Fy7ohmiKrIVOG2N0cDI5Ukcj1rKj8CO7Mv96l8vyR3Hj74YTWWsx3ngq0u9a8P3yNLBLYKbpYWDFWj3JuJAI4J9cZJUmvd9VhbQ/gPcWepmO2nt/Dn2WVXkXCy/Z9mzOcElsKMHkkYzmvnj4Z/Ey+8AaoUcSXOi3Dg3VoDyD08yPPAcDt0YDB6Ar9V38Nv4u8G3MNndbbXV9PdIbjyycJLGQr7Tg9GBwcfhWpwnxh4Ttft3jLQ7P7RPb+fqFvF51u+ySPdIo3I3ZhnIPY16x8efFOl+JNYs/DGmWM93q+nXZjNxG25SXVQYUVSd7FtoOQCrR4GcmvD66DwPr1n4Y8aaXrN/ZfbbW0l3vCACeQQGXPG5SQw6cqOR1AB6p4S/Z1u7y3iu/FOoSWBLsHsLUI8m3GFJlyVBzzgBuMcgnix8R/gp4Z8P+Gr7WdM1eSwnR2kit7+4XypBhm8mM4Db8A7cliduD13D3PRNb07xHo8GraTcfaLGfd5cuxk3bWKnhgCOQRyK0KAPgSAQtcRLcSSRwFwJHjQOyrnkhSQCcdsjPqK+559AsZvCsvhxFkg097I2KrG2WSIpswC2eQvc59818efEDwdceB/FtzpMvzW7Zms5C4YyQFiELYAw3BB4HIOOME+rv8U4tR+Bd1BJq9jJrsenmxurS4V/Om3yCISKzMN7eVvZsbvmYE7QNpAPEL+zt5PENzZaGZ721a7eKxOwmSZC5Efy4BLEY4wOT0r6j+EHwzbwLpc19qZjfWr5FEqqFYWyDnyw3UknBYg4JCgZ27j4p8DvD1vr/wASrZ7pv3emxG/CYPzujKE5BGMM6t3ztwRg19b0AFcl4E/5mX/sPXX/ALLXW1yXgT/mZf8AsPXX/stZT+OPzO7D/wC61v8At38zraKKK1OEKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigArJ8U/8ihrX/XhP/6LatasnxT/AMihrX/XhP8A+i2qZ/CzfC/x4eq/MPC3/IoaL/14Qf8Aota8G8R/AfxZrXjLW7+3udKjtbu7kuonlncZEkjnbgISGAxnt8wwTzj3nwt/yKGi/wDXhB/6LWtaiHwoMV/Hn6v8z50039mnUpbdm1TxJaW0+8hUtbZp1K4HJZihBznjHYc88U7H9m7xJJeRrf6zpUFqc75IDJK68HGFKqDzj+IevPSvpeiqMCvYWNvpmnW1hZx+Xa2sSQwpuJ2ooAUZPJwAOteX/Er4OXHjvXG1mHxB5EyWghhtZrYMgK7iBvUghSWychiMnqMKPWKKAPnDU/2a9Yi8r+yfEFjdZz5n2uF7fb0xjbvz364xgdc8d/oPwovNJ+FWp+EDr/lXWpSiaW7ghOIsiMPGBuBdSI2XJ25DcjtXqFFAHzpqX7NOpRW6tpfiS0uZ94DJdWzQKFweQylyTnHGO55453/hd8IPE3gjxiNXvdV037Ibd4ZYrXdI0wbGFO5F2gMA2Qc/KBjBNe2VXvr+z0yzkvL+7gtLWPG+aeQRouSAMseBkkD8aALFfInxvn025+Kmpvp8skrhIku2LKyCZUCkJjsFCgg8hg49K9v8R/HLwfpFheNpt9HquoQOES0jEiLKd4VsS7CmAMnIyDjjrXjHwm0K+8cfFJNYvHkZLO4/tO8uFXbul37lXhdoLPzt4+VXxjFAH1PpOmw6No1jpdu0jQWVvHbxtIQWKooUE4AGcD0Fc947/wCZa/7D1r/7NXW1yXjv/mWv+w9a/wDs1ZVvgZ3Zb/vUfn+TOtooorU4TkviZ/yT3VP+2X/o1K4z9o7/AJJ5p/8A2FY//RUtdn8TP+Se6p/2y/8ARqVwn7SM1wvg3SYVtd1q+ob5LjzAPLcRuFTb1O4M5z22e4rJfxX6L9Tuqf7jT/xz/KmeGeBvDGpeLPEsdho+o2lhqEaG4gkuJ2iJZCDhCoJ3j73HZSe1eieIdX+NNrZjwy9rqrrZfum1HTbWR3u1yCjeeoycADldrHJD5Oa5f4Jf8le0L/t4/wDSeSvr+tThPmD4b/BHVNb1GK/8U2U+n6RH8/2eX93NckEjZt+9GvHJOCQRt67l978Yabq7+A7/AE7wk0dlqAt1jsxEREEUEZRDjCEoCqnjBI5XGR0lFAHyh/wpj4la9rHm6tbbZpv9ZfX9+kvReNxVnc8AKMA9ugr0P4J+EPHHhDWb+31y0ktdFnty4jNxFIv2gMgBAViQdu4E8A4Gegr2yigD448dfDHXfB+uGBbGe4026uzBp08ZEpmzyiEKMiQg4xgZIbbkDNanhH4c+MviG9pBql1qVnotpbhre5v1kdERlGxYEYgMCFX7pAAA5+6D9Z0UAfFHj/wXceA/FD6PPP8AaYzEk0FxsCeajDBO0M23DBl5P8OehFe7/AHwfqnhzQ9V1DVraezm1CVEjtZ4tjqkW75yCcjJdhggfdzyGFewUUAeL/FP4Kf8JHeSa74YEEGpSbnurRzsS5bBO5T0WQng5wrZySDkt4xpmhfEfRPN/snSvFdh52PM+yW9xFvxnGdoGcZPX1NfZ9FAHy54X+Dfi/xpqiar4pku7K0kdTPNfSMbuZRlcKrZIPygZfGAQQGAxXR/Fex+Immaxp9h4Wj1WPw5a2iQ2KaK0xZQqqGExX5iwwMZ424xzvr6AooA+FNS8Na9o1utxqmialYwM4RZLq1eJS2CcAsAM4BOPY19N/BS68azaHqFv4ut75Y4JVFnNqCMs75yXB3/ADMo+XBI/iIBIGF9QooA+dPir8FbuLVDq3g3TJJrSZJJbqzhKAW7Lg/u1JBYNk4RQcEEDghRwfhT4seKfB2h3GkabPBJayZMP2mMyG1Y5yY+cDJOcMCuRnHLZ+x6jjghheZ4oo0eZ98rKoBdtoXLep2qoyewA7UAfHGqweP/AB1pdx4o1KLUtQ0+zTebh12xIv3WMSDAIGz5ig425bHWuf0DxHq/hbVF1LRb6S0uwhTeoDBlPUMrAhh0OCDyAeoFfddFAHnXwYvrjU/Btzf3knmXV1fNNM+0Dc7RxljgcDJJ6V6LXJeBP+Zl/wCw9df+y11tZUfgR3Zl/vUvl+SCvnP4teNNd8K+Mtas9Fu/sf8AaPlCeZFHmBVgUAKx+7/rCcj5gQMEc5+jK5Lw/wD8lC8Y/wDbl/6KNFT4o+v6MMJ/Br/4F/6XA+NbC+uNM1G2v7OTy7q1lSaF9oO11IKnB4OCB1r6D8cfFLxXa/DLwtrdhZyaVeao7tcTiIOiBQQqgSIQBID5i85wvBYZNe6UVqcJ8KWek694luLq4stP1LVZ9++5khhedtzEnLkAnJIJyeuDX1H8JPEPjXW7PUYPGOmz20lr5f2eeeya3efcXLZBAU7cKPlA685JzXpFc3488Vw+DPB1/rLmMzomy1jfH7yZuEGMgkZ+YgHO1WI6UAfOnxburzQfjpc6vHb4khltLy285Dsl2Rx4PbK7kIOD2IzkV1Fj+0veR2ca3/hiCe6Gd8kF4Ykbk4wpRiOMfxH146Vyfwz8CzfFXxBqN5rur3bQWiRm5lMpe4mZgyoAzggABDknPAAA5yPU/wDhnHwf/wBBLXP+/wDD/wDGqAMvTf2ltNluGXVPDd3bQbCVe1uVnYtkcFWCADGec9hxzx6p4T8aaF4105rzRbvzfK2ieF1KSQswyAyn8RkZUkHBODXlepfs06bLcK2l+JLu2g2AMl1bLOxbJ5DKUAGMcY7Hnnjyz4Q6pcaV8UNFe3OVuJTbSobgQq6OCOSeGwcMF/iZVA5IoA2Pjp4i1TWPHL2F5aT2lhp25LJJ7fy2kBIDygn7ysyHaRxtUcA5zX+FvxJ8QeEZ20bTNI/tm1upWmNjGjecXCYJRlBPRVJyrcJxjJNfW9FAHjfxl+J2teE3bQtP0mSAXtuQmqyyEAgqQ3k7CCHUlTuJyD/Dgqx+bLC+uNM1G2v7OTy7q1lSaF9oO11IKnB4OCB1r73qOeCG6t5be4ijmglQpJHIoZXUjBBB4II4xQBn+G9XbxB4a03WHs5LM3tuk/kO6uVDDI5XggjkdDgjIByB8WeKPDt54W8Q3mk3kU6+TK6wyzQmLz4w7Ksig/wttJBBI9zX3PUc8EN1by29xFHNBKhSSORQyupGCCDwQRxigD5k0T47/EC48jTbbT7HV75t20/YpGml6sfliZRwPRRwOe5qP45+ONX1fxBN4VubGOxs9MuN+wSCRp2wfLlLAfKDG4IQdNxyScY+k9M0LR9E83+ydKsbDzseZ9kt0i34zjO0DOMnr6mtCgD4w+H/AMRNU+H2ozzWMMFza3ewXVvMMbwpJBVhyrYLAHkfNyDgV7/8UvifceEvCWmzadYzwanrMRa3+1xhTaAKpYuhz+8G8AL0znOcYb1Cq99YWep2clnf2kF3ayY3wzxiRGwQRlTwcEA/hQB8MaFqf9ieIdM1byfO+w3cVz5W7bv2OG25wcZxjODX1H4r+MNjofgPT9etdPuxeaujnTrS9i2HCnBkkwSNmCrAA5YMuMAkr1n/AAgng/8A6FTQ/wDwXQ//ABNSeJfB+geMLeCDXtNjvEgcvES7IyEjBwykHB4yM4OB6CgD4cr6r8M/GzTdV8Ealruo6ddxT6QkAv47dVZXaVyimLcwOMjJDYxnGWxk9RqXwx8EarbrBceGNNRFcODawi3bOCOWj2kjnpnHT0FXNB8D+GvDWnXlhpOkQQ2t7xco5aXzhjG1i5JK4J+XpyeOTkA+PPGHiWbxh4sv9ent47d7t1IhQkhFVQijJ6naoyeMnPA6V7n8BPiBZ3WmW3gma3+z3VpFJLbzmYEXOZGdlCnBDAPkAbshWPGK9Q/4QTwf/wBCpof/AILof/iaj03wB4U0fxA2u6dodpa6gUKCSIFVQEAHYmdiEgYJUAnJ9TkA+bPi38SrP4hXmnLYafPbWun+Zskncb5fMCZyoyFwUP8AEc5zx0rY+AXjHS/DmuXmk6h56zazLbw20iJuRXHmABucjJdQMA9ecDmvoOPwX4VhSZIvDWjIkybJVWwiAddwbDfLyNyqcHuAe1SWPhPw3pl5HeWHh/SrS6jzsmgso43XIIOGAyMgkfjQB5n+0bPqUPgrT0t5Y00+a9CXShmDu20tGOOCnysSD3CEdDXmnw48aeA/Dnh64s/FHhn+1L57tpUm+wQT7YyiALukYEchjjpz710nxi8XzeOvEFv4B8OafJdyW16fMk2kNJcKGUqoOAEUF9zHjjPCrlsv/hnHxh/0EtD/AO/83/xqgD0Pw18XPhXZW88tlbx+H3kcLJENM2NKFHDHyQwI+YgZOevHPPUab8WvAeq3DQW/iW0R1QuTdK9uuMgcNIqgnnpnPX0NeKf8M4+MP+glof8A3/m/+NV5nr/hzV/C2qNputWMlpdhA+xiGDKehVlJDDqMgnkEdQaAPr7wJ/zMv/Yeuv8A2WutrzL4Ff8AJPP+3n/2lFXptZUfgR3Zl/vUvl+SCvGvix4OuPG3iWawseb+30Zbq1QuEWR1nIKkkd1ZsdPm25IGa9lrktZ0bxD/AMJemuaG+l/8eAs2S+Mn/PQuSAg+nf14oq3sml1DAOPNOMpJXi1rtc+XfAHiu4+HfjlL66tpxGm+1v7XaFk2E/MuGHDKyq2OMlcEgE19NyfFrwHFpcOot4ltDBK+xUVXaUHn70QXeo+U8lQOnqM8d4x+E+teNrz7dfweHra/O0Pd2ck6PIqggBgQVPUc43fKozgYrmP+GcdT/wCglaf9/wBv/jVHtfJh9R/6eQ+88kv5rjxd4yuZrO123Wr6g7w2/mA4eWQlU3HA6sBk4/CvtvSdNh0bRrHS7dpGgsreO3jaQgsVRQoJwAM4HoK868GeBvFPgXS3sdHi8NkyvvmuJzO0sx5xuIAGADgAAAcnqST0n/Fw/wDqV/8AyYo9r5MPqP8A08h951tfNH7QPgv+ytcg8TWNpBDYX+Irkxtgm6+ZixX/AGlHUdSrE4Jy3tX/ABcP/qV//Jio54PHt1by29xF4UmglQpJHIs7K6kYIIPBBHGKPa+TD6j/ANPIfeeA/C34vXHgfdpmqpPe6E25kjjwZLZzzlMkAqT1UkcncOchvU/FHx/8L6dpbnw/JJquoOjCIGB44o24wZC4Ukck4XOduCVyDXJz/s66jNcSypd2ECO5ZYo7iQqgJ+6N0ZOB05JPqTRB+zrqMNxFK93YTojhmikuJArgH7p2xg4PTgg+hFHtfJh9R/6eQ+8xPhd4dvvHHjG58ceItUkjtNKuEvLi7m+XzZU+cLuI2KiBQW/urtAABBFP4zfETS/Heo6bDpEM/wBl07zh9omG3zi5UZVeoXCAgnBO7kDHPuX9ieMv+Ee/sD7P4X/sz7J9i8jfdf6nZs27s7vu8Zzn3rgNT+ANzf8AlfZrfRdN2Z3fZLq5bzM4xnzQ/THbHU5zxg9r5MPqP/TyH3nmnwq8b2fgLxbJqd/az3FrNaPbOICN6ZZWDAEgHlAMZHXPbB+ivGOt6d4j+Et7q2k3H2ixn2eXLsZN22dVPDAEcgjkV5pB+zrqMNxFK93YTojhmikuJArgH7p2xg4PTgg+hFd/ceEfFknhIeGbeHwzZ6YqKiJbGcFQrBupBySRkk5JySeTmoqzcoOKi9UdWCw0aOKp1Z1Y2jJN69E7npNFFFdB45k+Kf8AkUNa/wCvCf8A9FtXg/xusbiTwB4Cv1jzaw2vkyPuHyu8URUY68iN/wAvcV7x4p/5FDWv+vCf/wBFtVPS9Ksdc+Hmm6ZqdtHc2dxpsCSxP0YbF/EEHBBHIIBGCKy/5e/I7v8AmB/7f/Q+Vvhh44h8A+Kn1S4sZLuCe3NrIscgVkVnRi4yMMQE+7kZz1Fe/wCkfHfwRqiXbz3N3pgt0V/9NhGZQW2/IIy5YgkZHXBzyAxHJ337NFnJeSNYeJ54LU42Rz2YldeBnLB1B5z/AAj05613mhfCHwpofh/U9FEN3e2mpvG119qnIZxGdyKDHtwA2TxycnJIwBqcJ4B8Z/FukeMfGsN7os0k9pBZR2/nNGUDsGdyVDYOPnA5A5B7YJ7/APZ68W6RZ6afCs00g1S9vZ7iFBGSu1Yo+C3QEhXI/wBw5xlc7n/DOPg//oJa5/3/AIf/AI1VPw3+z1DofiXTdWn8SyXSWVwlwIUshEXZDuUbi7YG4DPHIyOOoAMj49+O9A1rRovDmnXUlxqFnqZa5UQsqxGNXRlJYDJ3NxtyPlPI4zy/wT+IGkeCdU1O31rzIbTUUjP2pVLiJo9+AyqCSG3kZGcEDjBJHofjL4A2/iHxRLq+max9gjvZTLdwywmXa7BizodwzltvynGMsQcALWfffs0Wcl5I1h4nngtTjZHPZiV14GcsHUHnP8I9OetAGR+0jo14niHSdc2brGW0+x71BOyRHd8McYGQ/HOTtbjiq/wX+KWheD9JudD1xZ7eOe7a5S8RDIi5jAKso+YfcGCA2d3OAMn2/RPAul6X8P4PB94P7SsFiaObz1x5hZi7EAfd+Ykrg5XA5yM15Jr/AOzZMHaTw5rkbIXAEGoqVKrt5PmIDuO7oNg4PXjkA6/X/j54P03S2n0eeTWLzeFW2WOSAY7szumAAPQE5I4xkjw/4Y/bLz4h2Ooz+fPuvofPuXy2ZHlVhuY/xNtY88nB9DXY6V+zfrzapbjWNV01NP35nNpI7S7fRQyAZPTJ6Zzg4wfUtb8OaR4W0vwxpui2MdpaDxBbvsUlizHdkszElj0GSTwAOgFZVvgZ3Zb/AL1H5/kz0GiiitThOS+Jn/JPdU/7Zf8Ao1K+La+2PiHa3F74F1K3tLeWed/K2xxIXZsSoTgDnoCa8C8dfDh9T1w3/hPw5rVlbz5ae0ubBkSJ/WPbu+U8/Lxt7cEBcHNRqO/ZfqepDD1K+CgqavaU76rqof5Hrl/8TvBX/Ctbm906+0rb/Z7pb6TcBd27aUWF4Ac7c4UgfLt5zt5r5Qv7641PUbm/vJPMurqV5pn2gbnYkscDgZJPSu/0D4VakdUU+I9K11dPCEkadZM0rN2GXUBR3J56YxzkbHi34X2l5cS3fhbRPElgCihLC6013j3ZwxEu8sBjnBDc55APFe2h3Mf7NxX8v4r/ADO4+HHxo8Pz6d4c8LXVpfW1+IobAS7FeEuoCJ8wO75sL/DwW5OBur2SeCG6t5be4ijmglQpJHIoZXUjBBB4II4xXzv8N/BNv4V1GLWdf8O+IdQ1KLmCGLTCYYHBOHyzAu2NpGQNpzwSAw9f/wCE7/6lTxR/4Lv/ALKj20O4f2biv5fxX+Z8k+MdAbwt4x1XRWWQJa3DLF5jKzNEfmjYleMlCp7deg6V9F/AHww+i+BpNUuYPLutWl81SdwYwKMR5U8DJLsCOqupyeMc38RPCdn471yLV4dJ8UaZdeUIrjGiCUTY+6xxIp3AcEknICjjHPp//Cd/9Sp4o/8ABd/9lR7aHcP7NxX8v4r/ADPn347+GLfw/wCPvtVlBPHb6rEbt2fJQzl28wKT/wABYjJxv7AgDrP2dfCLG4vvFV7ZSBFQW+nyyKu1iSfNZcjORgLuGB8zjnnG18TNOh+ItvpytpHivT57F5CjjSRKrK4XcCu9TnKLg59eDnjqPDmt2Phbw/Z6LpvhPxWLS1QqnmWO5mJJZmJ3dSxJ4wOeABxR7aHcP7NxX8v4r/M9Br4EnnmuriW4uJZJp5XLySSMWZ2JySSeSSec19nf8J3/ANSp4o/8F3/2VeIfEjwNeeJfFEuteHtA1q2+2fPdQXOnGNVkwBuQoDnd1Oed2Tk7sA9tDuH9m4r+X8V/memf8Lv8A2fh7fp13tmhtM2+m/ZZIsME+WHIQovIC5BKj3FfLmralNrOs32qXCxrPe3ElxIsYIUM7FiBkk4yfU16f4O+GMVpefavF2g+Ib2NdwWys7Fwj5AwzSblbj5vlAHRTuxkVoeNPh1Z69rF1qmh6L4h0rzIhtsBoYEPmKuBgo42KcDPyscljznFHtodw/s3Ffy/iv8AM9A+FvxX03xLpcGkXi3cGoabpnnXt3dSKYmWLajSGQtnJyGORxzycZPgHxL1+x8UfEPV9Y0xpGs53jWJ3XaXCRqm7HUAlSRnBwRkA8V6Pofw30Kw0fULPUtK8bXdxebU+021gbby41ZW27N7K2WUE7sjhcAEZPP/APCn/wDpp4o/8Jz/AO30e2h3D+zcV/L+K/zPQPhPP4c8cfCR/At7LJ9ogST7TAG2PtaYyJLGe4Viv0IwwwRu+fL+1vPDniG5s/tGy+027eLzrdyNskbkbkbgjkZB4Nex+APBP/CHeL0125sPFF59l3i1ji0fy926PaWfLnGNzjaPRTu6rWx488KaF42vJtTTw54v07V5du+5TTzIkmAqjdGX7KuBtK9cnNHtodw/s3Ffy/iv8zqPht8XNI8W2Wm6VfXEkXiR0MUkLREidkTc0isq7AGAJwcYOQARgn5k8VeHrjwp4o1HQ7pt8lpKUD4A8xCMo+ATjcpU4zxnB5r1fwj8OrHw54ltNYv7DxXqItHE0MC6N5A81SCrMfNYkAjOOMnGTjIPT/EzTrHx/pYdPCviS21q3Qi1uzpvBHXy5MNkoT36qTkdSGPbQ7h/ZuK/l/Ff5nW+F/i94Q8R6WlxNq1ppd2EUz2t9OsRjY54VmwHHBOV7EZCk4rl/iD8ddI0zS5rPwneR3+rO7RfaBGTFbY4LgsMSH+7jK9ySAA3i/8Awqvxh/0Brv8A8BZv/iKkg+FXipriJbjSr+OAuBI8dlK7KueSFKgE47ZGfUUe2h3D+zcV/L+K/wAz1j4HWt54N8Dax4h8SXH2DRLny7i3E7nhQGDSBO2/KBcfM+0YBG3PgGu6n/bfiHU9W8nyft13Lc+Vu3bN7ltucDOM4zgV9FfEqyt/iJp1jC2heKLG6s5WeO4/sky/IwwybfMUckIc9tvua84/4U//ANNPFH/hOf8A2+j20O4f2biv5fxX+Z7h8PviD4U1/wANQwabPHp502yUz2NxIQbWJBtzvbh0AUfPngFd20nFeGfG7xT4a8XeIdNv9AvJ7uSO0MNw7RNHGAHJQKGAbd8z57YK45zV/wAI/Du50rVNTbWLLxWbCa3ktYl020eF50fIzLzgADadnzAnGSQMNz+pfCfXorhV0ux1a5g2As91pjwMGyeAqlwRjHOe5445PbQ7h/ZuK/l/Ff5na/s6eI9N0+41nSb++tLae7eBrRZQqNM2WUqHxljlkwhPclRy1fRdfI/hz4X6rD4gs5fEWgatPpKOWuIrS2kEjgA4UZUcFsA8g4zgg4r6Hn8Zw3VvLb3Hg/xJNBKhSSOTTAyupGCCC2CCOMUe2h3D+zcV/L+K/wAzZ8VeHrfxX4X1HQ7ptkd3EUD4J8twco+ARnawU4zzjB4r4g/0zS9R/wCW9nfWsvvHJDIp/AqwI+oIr7K/4Tv/AKlTxR/4Lv8A7KvOPiR4c0vxz5upW3hfxRY66ItqTDTv3c5GNolGcnAG0MOQCM7goWj20O4f2biv5fxX+Z0HgT426Br2l20XiDULTTNaZ3WSNkaKAgZKsrsSoG3A+ZgdwIA5GfNPjd8R9G8Yf2bpuhiC7tbfM7X7ROkiucqYlDqCFwAx67jt6beeS/4VX4w/6A13/wCAs3/xFXLP4T689vdNe2OrQzqmbZIdMeVZGweHYlSgzjkBup445PbQ7h/ZuK/l/Ff5lT4XeNYfAfjEapdwSTWctu9vcLEgaQKcMCmWUZ3Imcnpnvivpb4bX1vqena5f2cnmWt1rM80L7SNyMEKnB5GQR1r5sg+FXipriJbjSr+OAuBI8dlK7KueSFKgE47ZGfUV9F/CTSbjRPCU9lcQXUWy8YR/aoDC7oI4wGKknGcep78mp54ynG3mbrDVaGFq+0Vr8vVdzvaKKK3PKPjP4qf8lD1b/r5m/8ARr17p8LfGmj+M/h+2leJLuC9vtPiZtQTU1QrJCrbklO7hlUbQWPIZct1BPJ6r8Irzx94h1/VrXVoLXyb67txFLEW3SK4ZOQeFO9gTgldowGzxif8M4+MP+glof8A3/m/+NVlR+BHdmX+9S+X5I8z8Sw2Nt4q1eDSzGdPjvZktTHJvUxByEw2TuG3HOTmvpfVvEekaF8AtPt9Svo4J9R8OLbWkZBZpZDagAAAE4yQCx4GRkjIrk/h/wDAGWOW21bxdL5ckUoddLjCSBtrAjzW+ZWVsMCgHQj5gcitz4o/BnUvGfiA65petRid0SJrW+LeXGqg/wCrZQSBnB27erMc84rU4T500LU/7E8Q6Zq3k+d9hu4rnyt23fscNtzg4zjGcGvVPjX4KsYzB408MCO60m+eQ309rL50azmQnzC+48MzFcABVKAdWAqP/hnHxh/0EtD/AO/83/xqva/CXgGHSPhvF4R12WPVYHRvPjZAI13NuKoQA2AxJDH5s8jbwFAPBPhP8VbjwdqMOmavdTyeHH3L5YAb7K7EHzBxuKjByoP8TMATwfq+vA5/2ZoWuJWt/FckcBcmNJLAOyrngFhIATjvgZ9BXonwq8EXngLwlJpl/dQXF1NdvcuYAdiZVVCgkAnhAc4HXHbJAOP/AGhvCf8AaPhy38TQvBHJpf7u4BT55o5HVVww/useFPHzscg8H5or7H+L2p2el/C/WmvIYJ/tEQtoYZmA3SOQFZcg5ZOZABz8meMZHzx8GPDf/CR/Eqw3SbIdN/4mEmGwzeWy7QODn5ymRx8u7nOKAPf/AISeAP8AhBfC/wDpke3Wb/bJe4l3quC2xBjj5QxzjOWLckYxsfETxJeeEfAmpa5YRwSXVr5WxJ1JQ7pUQ5AIPRj3rqK5vx94am8X+CNT0K3uI7ee5RDHJICV3I6uAccgErjPOM5wcYoA+MJtW1K51QapPqF3LqAdXF28zNKGXG07yc5GBg54wK+ovgpr7axo2pRXTSSagZkvrmUqqq5nXqMd90bkjAAyMe3zzffDjxrp95JazeF9VeRMZMFs0yHIB4dAVPXseOnWvor4K6J/ZHh6/wDtNv5Wppciyuvn3Y8lBheCV4Z5OR1z1PFZT+OPzO7D/wC61v8At38z02iiitThCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKyfFP/Ioa1/14T/8Aotq1qKUldNGlKfs5xn2aZwOgfEPwtZeHNLtLjVNk8FpFHIv2eU7WVACMhcdRWj/wszwh/wBBf/yWl/8AiK62islGola6+7/gnbUr4Kc3N05a6/Gv/kDkv+FmeEP+gv8A+S0v/wARR/wszwh/0F//ACWl/wDiK62inar3X3f8Ej2mB/59z/8AA1/8rOS/4WZ4Q/6C/wD5LS//ABFH/CzPCH/QX/8AJaX/AOIrraKLVe6+7/gh7TA/8+5/+Br/AOVnJf8ACzPCH/QX/wDJaX/4ij/hZnhD/oL/APktL/8AEV1tFFqvdfd/wQ9pgf8An3P/AMDX/wArOS/4WZ4Q/wCgv/5LS/8AxFeUfHHxJF4u07R7Dw9d/a7WOWSa6TyzHhwFEZy4B6NJ09ee1fQlFFqvdfd/wQ9pgf8An3P/AMDX/wArPheDw1qU1xFE6RwI7hWlkcFUBP3jtycDrwCfQGvp/wAI+JPAPg7w1aaPYanGBEgM0q2cymeXA3SN8pOSR0ycDAHAFek0UWq9193/AAQ9pgf+fc//AANf/Kzkv+FmeEP+gv8A+S0v/wARWH4i8YaD4gvPDtppd99onTWraRl8l0woJGcsoHUivSaKUoVJKza+7/gmtHFYSjNVIU5XXeat/wCkIKKKK2PMOS+Jn/JPdU/7Zf8Ao1K+WfHHxD1rx/cWcurJaRJZoywxWsZVQWI3MdxJJO1R1x8owBzn7I1PTLPWdOlsL+HzrWXG9NxXOCCOQQeoFc7/AMKz8If9Aj/yZl/+LrJqam5RS2XX18n3PQp1MNPDRpVZSTUpPSKe6iuso/ynx7omt6j4c1iDVtJuPs99Bu8uXYr7dylTwwIPBI5FfR/wl+LWseO/EN7pOrWFjF5VobmOW0Dpja6qVIZmzneDkEYweueOz/4Vn4Q/6BH/AJMy/wDxdH/Cs/CH/QI/8mZf/i6L1ey+/wD4BPs8D/z8n/4Av/lh1tFcl/wrPwh/0CP/ACZl/wDi6P8AhWfhD/oEf+TMv/xdF6vZff8A8APZ4H/n5P8A8AX/AMsOtorkv+FZ+EP+gR/5My//ABdH/Cs/CH/QI/8AJmX/AOLovV7L7/8AgB7PA/8APyf/AIAv/lh1tFcl/wAKz8If9Aj/AMmZf/i6P+FZ+EP+gR/5My//ABdF6vZff/wA9ngf+fk//AF/8sOtorkv+FZ+EP8AoEf+TMv/AMXR/wAKz8If9Aj/AMmZf/i6L1ey+/8A4AezwP8Az8n/AOAL/wCWHW0VyX/Cs/CH/QI/8mZf/i6P+FZ+EP8AoEf+TMv/AMXRer2X3/8AAD2eB/5+T/8AAF/8sOtorkv+FZ+EP+gR/wCTMv8A8XR/wrPwh/0CP/JmX/4ui9Xsvv8A+AHs8D/z8n/4Av8A5YdbRXJf8Kz8If8AQI/8mZf/AIuj/hWfhD/oEf8AkzL/APF0Xq9l9/8AwA9ngf8An5P/AMAX/wAsOtorkv8AhWfhD/oEf+TMv/xdH/Cs/CH/AECP/JmX/wCLovV7L7/+AHs8D/z8n/4Av/lh1tFcl/wrPwh/0CP/ACZl/wDi6P8AhWfhD/oEf+TMv/xdF6vZff8A8APZ4H/n5P8A8AX/AMsDwJ/zMv8A2Hrr/wBlrraztG0LTfD9m9ppdt9ngeQyMu9nyxAGcsSegFaNVTi4xSZnja0K1eVSGz777fMK5Lw//wAlC8Y/9uX/AKKNdbXAjU7jw/468R3EmhazeQXn2bypLK0MinZFg85A6nHGehqKrScW+/6M3wEJVIVoR3cdP/A4M76iuS/4Tv8A6lTxR/4Lv/sqP+E7/wCpU8Uf+C7/AOyp+2h3I/s3Ffy/iv8AM62vN/jd4Y1TxR4BEOkQfaLizu1u2gX78iKjqQg/ib5wcd8HGTgHb/4Tv/qVPFH/AILv/sqP+E7/AOpU8Uf+C7/7Kj20O4f2biv5fxX+Z82/Cz4lf8K81G88/T/tljf+UJ9j7ZI9hOGXPDcO3ynGTj5hzn3uD44/D6a3ilfWpIHdAzRSWcxZCR907UIyOnBI9Ca8jn+EmmtbyrbweMo5zcFo3k0ZXVYccIVDgl8/x5AP90Vy8/wq8VLcSrb6VfyQByI3kspUZlzwSoUgHHbJx6mj20O4f2biv5fxX+Z6Z8QPj/D9n+weCZJDOXBk1KWABQuAcRo4ySTkEsoxg4ByGGJ+z94L/tXXJ/E19aQTWFhmK2MjZIuvlYMF/wBlT1PQspGSMrlaJ8Jf9RLr9r4o/i863sNJ+u3bK7f7pOU9R717npXiax0PS7fTNM8F+JLazt02RRJp3Cj/AL6ySTkknkkknJNHtodw/s3Ffy/iv8zuaK5L/hO/+pU8Uf8Agu/+yo/4Tv8A6lTxR/4Lv/sqPbQ7h/ZuK/l/Ff5nW0VyX/Cd/wDUqeKP/Bd/9lR/wnf/AFKnij/wXf8A2VHtodw/s3Ffy/iv8zraK5L/AITv/qVPFH/gu/8AsqP+E7/6lTxR/wCC7/7Kj20O4f2biv5fxX+Z1tFcl/wnf/UqeKP/AAXf/ZUf8J3/ANSp4o/8F3/2VHtodw/s3Ffy/iv8zraK5L/hO/8AqVPFH/gu/wDsqP8AhO/+pU8Uf+C7/wCyo9tDuH9m4r+X8V/mdbRXJf8ACd/9Sp4o/wDBd/8AZUf8J3/1Knij/wAF3/2VHtodw/s3Ffy/iv8AM62iuS/4Tv8A6lTxR/4Lv/sqP+E7/wCpU8Uf+C7/AOyo9tDuH9m4r+X8V/mdbRXJf8J3/wBSp4o/8F3/ANlR/wAJ3/1Knij/AMF3/wBlR7aHcP7NxX8v4r/M62iuS/4Tv/qVPFH/AILv/sqP+E7/AOpU8Uf+C7/7Kj20O4f2biv5fxX+Z8sz6vrXhn4pS61qlnJa6tb6mbu6tY3MWSz73QNz8jKxGfmBVu4PP0/4X+KvhDxYiLaapHa3buqCzvisMpZmIUKCcOTjohbqM4JxXD/ETQtL+IEsV7JoHi+x1OGIQx3Cab5iFN27DIWGcZfGCv3uc4Arx/8A4VX4w/6A13/4Czf/ABFHtodw/s3Ffy/iv8z681nXNL8Pac9/q9/BZWq5G+Z8biATtUdWbAOFGSccCvkT4h+L5viR41W8s9PkRNiWdlbqpaWRdzFdwGcuzOeB0yBzjJT/AIVX4w/6A13/AOAs3/xFeofC3w5b+B92p6r4X8Q3uutuVJI9OJjtkPGEyQSxHViBwdo4yWPbQ7h/ZuK/l/Ff5nb/AAYsbjTPBtzYXkfl3VrfNDMm4Ha6xxhhkcHBB6V6LXJeAUuPsetXFxZXVp9q1ae4jjuojG+xgpBIP5fga62ij8CDMv8Aep/L8kFFFFanCFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAGT4p/5FDWv+vCf/wBFtR4W/wCRQ0X/AK8IP/Ra1Lr9rNe+HNUtLdN889pLHGuQNzMhAGTx1NcxpWpeL9M0iysP+EM8z7LBHDv/ALUiG7aoGcY4zisZS5al327Nnp0aTrYRwi1dSvrKMdLebR3FFcl/wkHi/wD6Ef8A8q0X+FH/AAkHi/8A6Ef/AMq0X+FP20fP7n/kZf2dW7w/8GQ/+SOtrzP4LalDrOjeJ9Ut1kWC98R3VxGsgAYK6xsAcEjOD6mr+u3ni/W/D2p6T/whvk/brSW283+04m2b0K7scZxnOMis/wAEweL/AAd4QsdA/wCET+2fZfM/f/2jFHu3SM/3ecY3Y69qPbR8/uf+Qf2dW7w/8GQ/+SPTaK5L/hIPF/8A0I//AJVov8KP+Eg8X/8AQj/+VaL/AAo9tHz+5/5B/Z1bvD/wZD/5I62iuS/4SDxf/wBCP/5Vov8ACj/hIPF//Qj/APlWi/wo9tHz+5/5B/Z1bvD/AMGQ/wDkjra5Lx3/AMy1/wBh61/9mo/4SDxf/wBCP/5Vov8ACs7Uz4p8QXmjR3Hhf7DBaanBdyTf2hFLhVJz8owehz+FRUqKUbJP7n/kdWDwk6NdVJyjZX+3B9H2kd9RRRXQeOFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAcl4E/5mX/sPXX/ALLXW15t4d8YaD4fvPEdpql99nnfWrmRV8l3ypIGcqpHUGtz/hZnhD/oL/8AktL/APEVzUqtNQScl957WPy/F1MRKcKUmnbVRdtl5HW0VyX/AAszwh/0F/8AyWl/+Io/4WZ4Q/6C/wD5LS//ABFae3pfzL7zj/svHf8APmf/AIC/8jraK5L/AIWZ4Q/6C/8A5LS//EUf8LM8If8AQX/8lpf/AIij29L+ZfeH9l47/nzP/wABf+R1tcnr/wAS/B/hfVG0zWNajgvFQO0SwySlAem7YpAOOcHnBB6EUn/CzPCH/QX/APJaX/4ivlzxlpFzfeNdbvbAx3VpdXstxFMh2gq7FwMPggjdg8dQcZGCT29L+ZfeH9l47/nzP/wF/wCRo/Fj4iJ8QNctZLKGeDTLKIpBHOFDl2wXc7c4zhRjcfuZ4yRXu/wU8H3HhTwMH1G28jUtRlNxKjxBZI0wAiMc5OAC2DjaZCMA5r5/8F+D9PvNZWXxTqEen6fbujtEYnla6G4box5f3AVBBbORkYB5x9N/8LM8If8AQX/8lpf/AIij29L+ZfeH9l47/nzP/wABf+R1tFcl/wALM8If9Bf/AMlpf/iKP+FmeEP+gv8A+S0v/wARR7el/MvvD+y8d/z5n/4C/wDI62uS8Cf8zL/2Hrr/ANlo/wCFmeEP+gv/AOS0v/xFRfDq6hvbPX7u3ffBPrVxJG2CNysEIODz0NT7SEqkeV33OhYTEUMJVdam435d0118zsqKKK3PJCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAIzIwuEiEMhRkZjKCu1SCMKec5OSRgEfKckcZkoooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAqMyMLhIhDIUZGYygrtUgjCnnOTkkYBHynJHGZKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA/9k=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "\n",
        "# Setup: 100 users, 50 products\n",
        "class SimpleRec(nn.Module):\n",
        "    def __init__(self):\n",
        "        super().__init__()\n",
        "        self.user_layer = nn.Embedding(100, 10)\n",
        "        self.item_layer = nn.Embedding(50, 10)\n",
        "        self.out = nn.Linear(10, 1)\n",
        "    def forward(self, u, i):\n",
        "        return torch.sigmoid(self.out(self.user_layer(u) * self.item_layer(i)))\n",
        "\n",
        "model = SimpleRec()\n",
        "# Predict for User 1 on Item 5\n",
        "print(f\"✅ Project 2 Prediction: {model(torch.tensor([1]), torch.tensor([5])).item():.4f}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "e0_9aG0P3gRP",
        "outputId": "ea4bf8a8-022e-484b-81c5-145c39a73819"
      },
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Project 2 Prediction: 0.6169\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from transformers import pipeline\n",
        "\n",
        "# 'sentiment-analysis' is a built-in task that downloads its own model\n",
        "classifier = pipeline(\"sentiment-analysis\", device=-1) # device=-1 means use CPU\n",
        "\n",
        "text_list = [\"PyTorch is amazing!\", \"I am struggling with this code.\", \"Success feels great!\"]\n",
        "results = classifier(text_list)\n",
        "\n",
        "print(\"✅ Project 3 Results:\")\n",
        "for t, r in zip(text_list, results):\n",
        "    print(f\" - '{t}' is {r['label']}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lnANB2Qc4kjA",
        "outputId": "ab0ef68e-bbbf-4874-bba3-0e4c59f845e3"
      },
      "execution_count": 22,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "No model was supplied, defaulted to distilbert/distilbert-base-uncased-finetuned-sst-2-english and revision 714eb0f (https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).\n",
            "Using a pipeline without specifying a model name and revision in production is not recommended.\n",
            "Device set to use cpu\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "✅ Project 3 Results:\n",
            " - 'PyTorch is amazing!' is POSITIVE\n",
            " - 'I am struggling with this code.' is NEGATIVE\n",
            " - 'Success feels great!' is POSITIVE\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification\n",
        "import torch\n",
        "\n",
        "# 1. Load a model specialized in Legal Language\n",
        "# This model is better at understanding \"heretofore,\" \"indemnify,\" etc.\n",
        "model_name = \"saibo/legal-roberta-base\"\n",
        "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
        "model = AutoModelForSequenceClassification.from_pretrained(model_name)\n",
        "\n",
        "# 2. Create the Analysis function\n",
        "def analyze_legal_sentiment(text):\n",
        "    inputs = tokenizer(text, return_tensors=\"pt\", truncation=True, max_length=512)\n",
        "    with torch.no_grad():\n",
        "        outputs = model(**inputs)\n",
        "\n",
        "    # Convert output to probabilities\n",
        "    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)\n",
        "\n",
        "    # For legal, we often map these to: Neutral, Risk/Contention, or Compliance\n",
        "    # This is a simplified example\n",
        "    labels = [\"Neutral/Compliant\", \"Contested/High Risk\"]\n",
        "    prediction = torch.argmax(probs).item()\n",
        "\n",
        "    return {\n",
        "        \"label\": labels[prediction],\n",
        "        \"confidence\": f\"{probs[0][prediction].item():.4f}\",\n",
        "        \"text_preview\": text[:50] + \"...\"\n",
        "    }\n",
        "\n",
        "# Test it\n",
        "legal_clause = \"The party of the first part shall indemnify and hold harmless the second part against all claims.\"\n",
        "print(analyze_legal_sentiment(legal_clause))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yZJk2CJ98G67",
        "outputId": "ed674cd6-36cb-48cf-a55e-0f7579ac07f0"
      },
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at saibo/legal-roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']\n",
            "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "{'label': 'Neutral/Compliant', 'confidence': '0.5505', 'text_preview': 'The party of the first part shall indemnify and ho...'}\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from fastapi import FastAPI, Depends, HTTPException\n",
        "from pydantic import BaseModel\n",
        "\n",
        "app = FastAPI()\n",
        "\n",
        "class LegalText(BaseModel):\n",
        "    content: str\n",
        "\n",
        "@app.post(\"/analyze\")\n",
        "async def get_analysis(item: LegalText):\n",
        "    # Here you would add a check: \"Does this developer have a valid API Key?\"\n",
        "    try:\n",
        "        result = analyze_legal_sentiment(item.content)\n",
        "        return result\n",
        "    except Exception as e:\n",
        "        raise HTTPException(status_code=500, detail=str(e))\n",
        "\n",
        "# Run with: uvicorn filename:app --reload"
      ],
      "metadata": {
        "id": "TyP-huUW9UBd"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install torch transformers fastapi uvicorn\n",
        "!pip freeze > requirements.txt"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "uCUIbTtP9W1y",
        "outputId": "aedba219-4e73-40fc-d5d8-5bd681906190"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: torch in /usr/local/lib/python3.12/dist-packages (2.9.0+cu126)\n",
            "Requirement already satisfied: transformers in /usr/local/lib/python3.12/dist-packages (4.57.3)\n",
            "Requirement already satisfied: fastapi in /usr/local/lib/python3.12/dist-packages (0.123.10)\n",
            "Requirement already satisfied: uvicorn in /usr/local/lib/python3.12/dist-packages (0.38.0)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from torch) (3.20.0)\n",
            "Requirement already satisfied: typing-extensions>=4.10.0 in /usr/local/lib/python3.12/dist-packages (from torch) (4.15.0)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.12/dist-packages (from torch) (75.2.0)\n",
            "Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch) (1.14.0)\n",
            "Requirement already satisfied: networkx>=2.5.1 in /usr/local/lib/python3.12/dist-packages (from torch) (3.6.1)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch) (3.1.6)\n",
            "Requirement already satisfied: fsspec>=0.8.5 in /usr/local/lib/python3.12/dist-packages (from torch) (2025.3.0)\n",
            "Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch) (12.6.77)\n",
            "Requirement already satisfied: nvidia-cuda-runtime-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch) (12.6.77)\n",
            "Requirement already satisfied: nvidia-cuda-cupti-cu12==12.6.80 in /usr/local/lib/python3.12/dist-packages (from torch) (12.6.80)\n",
            "Requirement already satisfied: nvidia-cudnn-cu12==9.10.2.21 in /usr/local/lib/python3.12/dist-packages (from torch) (9.10.2.21)\n",
            "Requirement already satisfied: nvidia-cublas-cu12==12.6.4.1 in /usr/local/lib/python3.12/dist-packages (from torch) (12.6.4.1)\n",
            "Requirement already satisfied: nvidia-cufft-cu12==11.3.0.4 in /usr/local/lib/python3.12/dist-packages (from torch) (11.3.0.4)\n",
            "Requirement already satisfied: nvidia-curand-cu12==10.3.7.77 in /usr/local/lib/python3.12/dist-packages (from torch) (10.3.7.77)\n",
            "Requirement already satisfied: nvidia-cusolver-cu12==11.7.1.2 in /usr/local/lib/python3.12/dist-packages (from torch) (11.7.1.2)\n",
            "Requirement already satisfied: nvidia-cusparse-cu12==12.5.4.2 in /usr/local/lib/python3.12/dist-packages (from torch) (12.5.4.2)\n",
            "Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /usr/local/lib/python3.12/dist-packages (from torch) (0.7.1)\n",
            "Requirement already satisfied: nvidia-nccl-cu12==2.27.5 in /usr/local/lib/python3.12/dist-packages (from torch) (2.27.5)\n",
            "Requirement already satisfied: nvidia-nvshmem-cu12==3.3.20 in /usr/local/lib/python3.12/dist-packages (from torch) (3.3.20)\n",
            "Requirement already satisfied: nvidia-nvtx-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch) (12.6.77)\n",
            "Requirement already satisfied: nvidia-nvjitlink-cu12==12.6.85 in /usr/local/lib/python3.12/dist-packages (from torch) (12.6.85)\n",
            "Requirement already satisfied: nvidia-cufile-cu12==1.11.1.6 in /usr/local/lib/python3.12/dist-packages (from torch) (1.11.1.6)\n",
            "Requirement already satisfied: triton==3.5.0 in /usr/local/lib/python3.12/dist-packages (from torch) (3.5.0)\n",
            "Requirement already satisfied: huggingface-hub<1.0,>=0.34.0 in /usr/local/lib/python3.12/dist-packages (from transformers) (0.36.0)\n",
            "Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.12/dist-packages (from transformers) (2.0.2)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.12/dist-packages (from transformers) (25.0)\n",
            "Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.12/dist-packages (from transformers) (6.0.3)\n",
            "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.12/dist-packages (from transformers) (2025.11.3)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.12/dist-packages (from transformers) (2.32.4)\n",
            "Requirement already satisfied: tokenizers<=0.23.0,>=0.22.0 in /usr/local/lib/python3.12/dist-packages (from transformers) (0.22.1)\n",
            "Requirement already satisfied: safetensors>=0.4.3 in /usr/local/lib/python3.12/dist-packages (from transformers) (0.7.0)\n",
            "Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.12/dist-packages (from transformers) (4.67.1)\n",
            "Requirement already satisfied: starlette<0.51.0,>=0.40.0 in /usr/local/lib/python3.12/dist-packages (from fastapi) (0.50.0)\n",
            "Requirement already satisfied: pydantic!=1.8,!=1.8.1,!=2.0.0,!=2.0.1,!=2.1.0,<3.0.0,>=1.7.4 in /usr/local/lib/python3.12/dist-packages (from fastapi) (2.12.3)\n",
            "Requirement already satisfied: annotated-doc>=0.0.2 in /usr/local/lib/python3.12/dist-packages (from fastapi) (0.0.4)\n",
            "Requirement already satisfied: click>=7.0 in /usr/local/lib/python3.12/dist-packages (from uvicorn) (8.3.1)\n",
            "Requirement already satisfied: h11>=0.8 in /usr/local/lib/python3.12/dist-packages (from uvicorn) (0.16.0)\n",
            "Requirement already satisfied: hf-xet<2.0.0,>=1.1.3 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<1.0,>=0.34.0->transformers) (1.2.0)\n",
            "Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.12/dist-packages (from pydantic!=1.8,!=1.8.1,!=2.0.0,!=2.0.1,!=2.1.0,<3.0.0,>=1.7.4->fastapi) (0.7.0)\n",
            "Requirement already satisfied: pydantic-core==2.41.4 in /usr/local/lib/python3.12/dist-packages (from pydantic!=1.8,!=1.8.1,!=2.0.0,!=2.0.1,!=2.1.0,<3.0.0,>=1.7.4->fastapi) (2.41.4)\n",
            "Requirement already satisfied: typing-inspection>=0.4.2 in /usr/local/lib/python3.12/dist-packages (from pydantic!=1.8,!=1.8.1,!=2.0.0,!=2.0.1,!=2.1.0,<3.0.0,>=1.7.4->fastapi) (0.4.2)\n",
            "Requirement already satisfied: anyio<5,>=3.6.2 in /usr/local/lib/python3.12/dist-packages (from starlette<0.51.0,>=0.40.0->fastapi) (4.12.0)\n",
            "Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch) (1.3.0)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch) (3.0.3)\n",
            "Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests->transformers) (3.4.4)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests->transformers) (3.11)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests->transformers) (2.5.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests->transformers) (2025.11.12)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from fastapi import FastAPI, Security, HTTPException, status\n",
        "from fastapi.security import APIKeyHeader # Corrected import\n",
        "\n",
        "app = FastAPI()\n",
        "\n",
        "# 1. Setup API Key Security\n",
        "API_KEY_NAME = \"access_token\"\n",
        "api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False) # Corrected class name\n",
        "\n",
        "# This would normally be a database of paying users\n",
        "VALID_API_KEYS = [\"user_123_stripe_active\", \"beta_tester_pro\"]\n",
        "\n",
        "async def get_api_key(api_key: str = Security(api_key_header)):\n",
        "    if api_key in VALID_API_KEYS:\n",
        "        return api_key\n",
        "    raise HTTPException(\n",
        "        status_code=status.HTTP_403_FORBIDDEN,\n",
        "        detail=\"Invalid or missing API Key. Visit stripe.com to subscribe.\"\n",
        "    )\n",
        "\n",
        "@app.post(\"/analyze-legal\")\n",
        "async def analyze(text: str, api_key: str = Depends(get_api_key)):\n",
        "    # Run the PyTorch Legal-RoBERTa model here\n",
        "    result = analyze_legal_sentiment(text) # From the previous code\n",
        "    return {\n",
        "        \"status\": \"success\",\n",
        "        \"billing_unit\": 1,\n",
        "        \"data\": result\n",
        "    }"
      ],
      "metadata": {
        "id": "DQFYfeRu94nq"
      },
      "execution_count": 20,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "{\n",
        "  \"risk_level\": \"High\",\n",
        "  \"clause_type\": \"Indemnification\",\n",
        "  \"sentiment\": \"Aggressive\",\n",
        "  \"confidence\": 0.98\n",
        "}"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rm4J6FEo94Vf",
        "outputId": "6714739a-b7fa-4ea5-8109-1a775975e907"
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "{'risk_level': 'High',\n",
              " 'clause_type': 'Indemnification',\n",
              " 'sentiment': 'Aggressive',\n",
              " 'confidence': 0.98}"
            ]
          },
          "metadata": {},
          "execution_count": 21
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "31ac693c",
        "outputId": "69a6d792-d631-4479-b183-1207e05b4f6f"
      },
      "source": [
        "import uvicorn\n",
        "import nest_asyncio\n",
        "import threading\n",
        "\n",
        "nest_asyncio.apply()\n",
        "\n",
        "# Configuration for Uvicorn\n",
        "config = uvicorn.Config(app, host=\"0.0.0.0\", port=8000, loop=\"asyncio\")\n",
        "\n",
        "# Create a Uvicorn server instance\n",
        "server = uvicorn.Server(config)\n",
        "\n",
        "# Run the server in a separate thread\n",
        "def run_server():\n",
        "    server.run()\n",
        "\n",
        "# Start the server thread\n",
        "server_thread = threading.Thread(target=run_server)\n",
        "server_thread.start()\n",
        "\n",
        "print(\"FastAPI server started in a background thread! It will be accessible at http://0.0.0.0:8000\")\n",
        "print(\"To stop the server, you will need to interrupt the kernel (Runtime -> Interrupt execution).\")"
      ],
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "INFO:     Started server process [22451]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "FastAPI server started in a background thread! It will be accessible at http://0.0.0.0:8000\n",
            "To stop the server, you will need to interrupt the kernel (Runtime -> Interrupt execution).\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "INFO:     Waiting for application startup.\n",
            "INFO:     Application startup complete.\n",
            "ERROR:    [Errno 98] error while attempting to bind on address ('0.0.0.0', 8000): [errno 98] address already in use\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9a711b14"
      },
      "source": [
        "After running the cell above, your FastAPI application will be accessible at `http://0.0.0.0:8000`.\n",
        "\n",
        "To interact with it from outside Colab (e.g., from your local machine), you would typically use a tunneling service like `ngrok`. Let me know if you'd like instructions on how to set up `ngrok` for public access."
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "gCrL7vthCrEZ"
      },
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "gpuType": "T4",
      "provenance": [],
      "mount_file_id": "1yyUDzthRgSYEs4BzrQwmhSDM_5-1Koep",
      "authorship_tag": "ABX9TyPGKlBya63MrCMmV2Ii+4ot",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}