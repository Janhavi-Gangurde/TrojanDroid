def getMinimumCost(machineCount, finalMachineCount, shiftingCost):
    # Sort machineCount in descending order
    machineCount.sort(reverse=True)
    
    # Initialize current counts for the three final regions
    current0, current1, current2 = 0, 0, 0
    cost = 0
    
    # Distribute machines to meet finalMachineCount
    for machines in machineCount:
        if current0 < finalMachineCount[0]:
            current0 += machines
        elif current1 < finalMachineCount[1]:
            current1 += machines
        elif current2 < finalMachineCount[2]:
            current2 += machines
        else:
            # Excess machines, calculate cost to move or remove
            cost += 1
    
    # Calculate remaining excess machines cost
    if current0 > finalMachineCount[0]:
        cost += (current0 - finalMachineCount[0])
    if current1 > finalMachineCount[1]:
        cost += (current1 - finalMachineCount[1])
    if current2 > finalMachineCount[2]:
        cost += (current2 - finalMachineCount[2])
    
    return cost * shiftingCost

# Example usage:
machineCount = [4,2,4,5,3,3,4,4,5]
finalMachineCount = [2]
shiftingCost = 5

result = getMinimumCost(machineCount, finalMachineCount, shiftingCost)
print("Minimum cost:", result)
