import numpy as np



def APFD(is_fault, index_order) -> float:
    """
    Compute APFD from the index order of the misclassified samples.
    """
    ordered_faults = is_fault[index_order]
    fault_indexes = np.where(ordered_faults == True)[0]
    k = np.count_nonzero(is_fault)
    n = is_fault.shape[0]
    # The +1 comes from the fact that the first sample has index 0 but order 1
    sum_of_fault_orders = np.sum(fault_indexes + 1)
    return 1 - (sum_of_fault_orders / (k * n)) + (1 / (2 * n))


def TRC(is_fault, index_order, budget) -> float:
    selected = index_order[:budget]
    selected_fault = np.sum(is_fault[selected])
    return (selected_fault/min([np.sum(is_fault),budget]))