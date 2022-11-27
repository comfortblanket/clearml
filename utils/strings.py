
def str_find_all(string, substr, start=None, end=None, overlaps=False):
    inds = []
    i = 0 if start is None else start
    end = len(string) if end is None else end
    string = string[:end]

    while i != -1 and i < end:
        i = string.find(substr, i)
        
        if i != -1:
            inds.append(i)
            
            if overlaps:
                i += 1
            else:
                i += len(substr)
        
    return inds
