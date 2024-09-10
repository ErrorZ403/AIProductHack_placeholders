from typing import Any
from typing import Dict
from typing import List

import numpy as np


async def get_optim_k(
    query: str, collection: Any,
) -> List[Dict[str, Any]]:
    full_results = await collection.query(query, n_results=2)
    documents = []

    for record in full_results:
        documents.append(record['content'])

    return documents

    # derivatives = np.diff(distances)
    # max_derivative_index = np.argmax(derivatives)
    # k = max(max_derivative_index + 1, 50)

    # return documents[:k]
