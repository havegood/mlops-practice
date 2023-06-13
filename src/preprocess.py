import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

CAT_FEATURES = [
    "area_type",
    "city",
    "furnishing_status",
    "tenant_preferred",
    "point_of_contact",
]


def extract_floor(floor_info: str) -> int:
    """층수 컬럼에서 실제 층수만 추출합니다.

    현재 층수 정보는 'X out of Y'로 되어 있습니다.
    여기에서 X만 추출하여 정수로 반환합니다.
    Upper basement, Lower basement, Ground out 등은 모두 0층으로 변환합니다.

    Args:
        floor_info (str): 층수 정보
    """

    num = floor_info.split(" ")[0]
    num = num if num.isnumeric() else "0"
    num = int(num)

    return num


def floor_extractor(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """`extract_floor()` 함수를 `FunctionTransformer`에 사용하기 위한
    Wrapping function입니다.

    Args:
        df (pd.DataFrame): 데이터프레임
        col (str): `extract_floor()`를 적용할 컬럼명
            `Floor`만 사용해야 함

    Returns:
        pd.DataFrame: 컬럼 처리 후 데이터
    """
    df[col] = df[col].apply(lambda x: extract_floor(x))
    return df


# 1. 방의 크기는 제곱근을 적용함 (FunctionTransformer 사용)
# 2. 층수는 실제 층수를 추출하되 숫자가 아닌 Basement 등은 0층으로 표기함
# 3. 범주형 변수(CAT_FEATURES)는 타겟 인코딩 적용 (from category_encoders import TargetEncoder)
preprocess_pipeline = ColumnTransformer(
    transformers=[
        (
            "sqrt_transformer",  # transformer의 이름
            FunctionTransformer(np.sqrt),
            ["size"],
        ),  # 본래는 fit, transform이 있어야만 사용가능한데, FunctionTransformer가 그렇게 함수들을 만들어주는것
        (
            "floor_extractor",
            FunctionTransformer(floor_extractor, kw_args={"col": "floor"}),
            ["floor"],
        ),
        ("target_encoder", TargetEncoder(), CAT_FEATURES),
    ],
    remainder="passthrough",  # 기본값은 버림
    verbose_feature_names_out=False,  # 원래 이름을 그대로 사용하게하는 옵션
)
preprocess_pipeline.set_output(
    transform="pandas"
)  # 본래는 numpy ndarray가 반환되는데, pandas옵션을 주면 컬럼명을 포함한 데이터프레임이 반환된다
