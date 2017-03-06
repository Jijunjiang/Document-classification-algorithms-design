import re
import __builtin__
   
def load(infile):
    try:
        del __builtin__.labels, __builtin__.labelset, __builtin__.fileIDs, __builtin__.MISSLE, __builtin__.vocabulary, __builtin__.wordcounts.keys()[:]
    except AttributeError:
        pass
    
    __builtin__.labels = {}
    __builtin__.labelset = set()
    __builtin__.fileIDs = []
    
    __builtin__.MISSLE = {}
    __builtin__.vocabulary = set()
    __builtin__.wordcounts = {}

    f = open(infile)
    fileID = 0
    for line in f:
        __builtin__.fileIDs.append(fileID)
        __builtin__.wordcounts[fileID] = {}
        
        parts = line.split(" ")
        label = parts[-1].split(":")[1][:-1]
        
        __builtin__.labels[fileID] = label
        __builtin__.labelset = __builtin__.labelset.union(set([label]))

        for part in parts[:-1]:
            (word,count) = part.split(":")
            count = int(count)

            if "_" in word: continue

            __builtin__.MISSLE.setdefault(word,{})
            __builtin__.MISSLE[word][fileID] = count
            __builtin__.wordcounts[fileID][word] = count
            __builtin__.vocabulary.add(word)
        
        fileID += 1

    f.close()

def load_arff(infile):
    try:
        del __builtin__.labels, __builtin__.labelset, __builtin__.fileIDs, __builtin__.MISSLE, __builtin__.vocabulary, __builtin__.wordcounts.keys()[:]
    except AttributeError:
        pass
    
    __builtin__.labels = {}         # {fileID: label}
    __builtin__.labelset = set()    
    __builtin__.fileIDs = []        # [fileID]
    
    __builtin__.MISSLE = {}         # {word: fileID}
    __builtin__.vocabulary = set()  # (word)
    __builtin__.wordcounts = {}     # {fileID: {word: count}}

    f = open(infile)

    # Read word map (ATTRIBUTE section)
    vocab_map = {}
    class_label_id = None
    
    next_id = 0
    for line in f:
        parts = line.split()

        if len(parts) > 0 and parts[0] == '@ATTRIBUTE':
            
            word = parts[1]
            
            if word == 'CLASS_LABEL':   # Ignore CLASS_LABEL as an attribute field
                class_label_id = next_id
            else:
                vocab_map[next_id] = word
                
            next_id += 1
            
        elif line == '@DATA\n':
            break
    
    # Check that ATTRIBUTE section was formatted as expected
    if class_label_id is None:
        raise ValueError('Error: Unexpected arff format. No id for class label was found.')
    if len(vocab_map.keys()) == 0:
        raise ValueError('Error: Unexpected arff format. No vocabulary ATTRIBUTES found.')

        
    # Read files line by line
    fileID = 0
    for line in f:
        __builtin__.fileIDs.append(fileID)
        __builtin__.wordcounts[fileID] = {} 
        
        # Expecting format {word_id count, word_id count, ...} where at least
        #   one word_id must be class_label_id
        if len(line) < 3 or line[0] != '{' or line[-2] != '}':
            continue

        parts = line[1:-2].split(",")
                
        label = None
        for part in parts:
            (word_id,value) = part.split(" ") 
            word_id = int(word_id)
            
            if word_id == class_label_id:
                # Class label for this document
                if label is not None:
                    raise ValueError('Error: Unexpected arff format. Multiple class labels found for document %d.' % fileID )

                label = value

            else:
                # General word count
                word = vocab_map[word_id]
                count = int(value)
                
                __builtin__.MISSLE.setdefault(word,{})
                __builtin__.MISSLE[word][fileID] = count
                __builtin__.wordcounts[fileID][word] = count
                __builtin__.vocabulary.add(word)
        
        # Check that document was formatted as expected
        if label is not None:
            __builtin__.labels[fileID] = label
            __builtin__.labelset = __builtin__.labelset.union(set([label]))
        else:
            raise ValueError('Error: Unexpected arff format. No class label found for document %d.' % fileID )

        if len(__builtin__.wordcounts[fileID].keys()) == 0:
            print line
            raise ValueError('Error: Unexpected arff format. Document %d appears to have 0 words.' % fileID )


        fileID += 1

    f.close()
