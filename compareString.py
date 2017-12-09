def compareString(a, b, i, j, memo={}): 
    """
    returns the minimum number of changes needed, to get from a to b string.
    changes are either insertions, replacements, or deletions
    
    i, j are the last (excluded) index where the
    substring cut off at comparison, for a, b respectively

    a is generally the predicted string, and b is the correct string.
    """
    if i == 0:
        return j
    if j == 0:
        return i
    if (a,b,i,j) in memo:
        return memo[(a,b,i,j)] #already memorized, return it.

    
    if a[i-1] == b[j-1]: #last character match, recurse
        return compareString(a, b, i-1, j-1, memo)
    
    #replaced character in 'a'
    replacedCost = compareString(a, b, i-1,j-1, memo) + 1

    #inserted character in 'a'
    insertCost = compareString(a, b,i-1,j, memo) + 1

    #delete character in 'a'
    deleteCost = compareString(a, b,i,j-1, memo) + 1

    memo[(a,b,i,j)] = min(replacedCost, insertCost, deleteCost)

    return memo[(a,b,i,j)]


def fractionAccuracy(predicted,correct): #set one of these to be t
    """
    predicted, correct: 2 strings to compare
    returns the accuracy (difference from correct string) as a fraction from 0-1
    (note a difference can be an insertion, deletion, or replacement)
    this function calls on compareString
    """
    difference = compareString(predicted,correct,len(predicted),len(correct))
    print('number of differences', difference)
    return 1-difference/len(correct)
