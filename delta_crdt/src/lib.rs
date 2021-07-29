use apbf::APBF;
use rand::Rng;
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

type Dot<ID> = (ID, u64);

#[derive(Debug)]
pub struct DotContext<ID> {
    /// Compact causal context
    pub causal_context: HashMap<ID, u64>,
    pub dot_cloud: HashSet<Dot<ID>>,
}

impl<ID> Default for DotContext<ID>
where
    ID: Clone + Default + Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<ID> DotContext<ID>
where
    ID: Clone + Default + Eq + Hash,
{
    /// Checks if a given dot is in the dot context.
    pub fn dot_in(&self, dot: &Dot<ID>) -> bool {
        let mut res = false;
        // rid - replica ID | mic - monotonically increasing counter
        let (rid, mic) = dot;
        // First, the dot is searched in the map
        if let Some(last) = self.causal_context.get(rid) {
            if *mic <= *last {
                res = true;
            }
        }
        if !res {
            self.dot_cloud.contains(dot)
        } else {
            res
        }
    }

    pub fn compact(&mut self) {
        let mut must_compact = true;
        let mut dots_to_remove = HashSet::new();
        while must_compact {
            must_compact = false;
            for dot in self.dot_cloud.iter() {
                if dots_to_remove.contains(dot) {
                    continue;
                }
                let (id, counter) = dot;
                let dot_clone = dot.clone();
                // Search for elements with the same ID in the causal context
                if let Some(val) = self.causal_context.get_mut(id) {
                    let cur_val = *val;
                    if *counter == cur_val + 1 {
                        // Contiguous interval - allows for compaction
                        *val = cur_val + 1;
                        dots_to_remove.insert(dot_clone);
                        // Might affect previosly seen dots, so a new round of compaction should happen
                        must_compact = true;
                    } else if *counter <= cur_val {
                        // The dot is dominated and, as such, must be pruned
                        dots_to_remove.insert(dot_clone);
                        // As this affects no previous values that have been iterated
                        // over, the additional compaction flag remains unchanged.
                    }
                } else {
                    // No entry for the specified ID in the causal context
                    // If the dot's counter value is 1, compaction is possible
                    if *counter == 1 {
                        let id_clone = id.clone();
                        self.causal_context.insert(id_clone, *counter);
                        dots_to_remove.insert(dot_clone);
                        // ... as this might enable compaction for previously iterated values
                        must_compact = true;
                    }
                }
            }
            self.dot_cloud.retain(|d| !dots_to_remove.contains(d));
        }
    }

    /// Creates and inserts a new dot in the causal context for a replica identified by the `id`
    /// parameter. Returns a copy of the newly created dot.
    pub fn make_dot(&mut self, id: ID) -> Dot<ID> {
        if let Some(val) = self.causal_context.get_mut(&id) {
            let new_val = *val + 1;
            *val = new_val;
            (id, new_val)
        } else {
            let id_clone = id.clone();
            self.causal_context.insert(id_clone, 1);
            (id, 1)
        }
    }

    /// Inserts a given dot in the dot cloud. Also allows for compaction of the dot context
    /// after insertion.
    pub fn insert_dot(&mut self, dot: Dot<ID>, compact_now: bool) {
        self.dot_cloud.insert(dot);
        if compact_now {
            self.compact();
        }
    }

    /// Joins two dot contexts.
    pub fn join(&mut self, other: &Self) {
        for (oid, ocounter) in other.causal_context.iter() {
            // If the ID at `other` exists, the counter at `self` will be updated
            //  to the maximum value observed for that ID
            if let Some(scounter) = self.causal_context.get_mut(oid) {
                *scounter = std::cmp::max(*scounter, *ocounter);
            // If not, the entry is new at `self`, and should be saved for future insertion
            } else {
                self.causal_context.insert(oid.clone(), *ocounter);
            };
        }

        // Join dot clouds
        for (oid, ocounter) in other.dot_cloud.iter() {
            self.insert_dot((oid.clone(), *ocounter), false);
        }
        // After joining dot clouds, attempt to compact the dot context
        self.compact();
    }

    pub fn new() -> Self {
        Self {
            causal_context: HashMap::new(),
            dot_cloud: HashSet::new(),
        }
    }
}

pub struct DotKernel<ID, T> {
    /// Classic map of dots to values - DotFun
    pub dots_to_values: HashMap<Dot<ID>, T>,
    pub dot_context: DotContext<ID>,
    /// Probabilistic map of dots (in u64 format) to values - DotFun
    pub prob_dots_to_values: HashMap<u64, T>,
    /// Is in fact a set of tombstones
    pub prob_dot_context: APBF,
    /// Maps a probabilistic dot to its classic counterpart. For inspection only.
    pub pdots_to_cdots: HashMap<u64, Dot<ID>>,
}

impl<ID, T> Default for DotKernel<ID, T>
where
    ID: Clone + Default + Eq + Hash,
    T: Clone + Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<ID, T> DotKernel<ID, T>
where
    ID: Clone + Default + Eq + Hash,
    T: Clone + Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            dots_to_values: HashMap::new(),
            dot_context: DotContext::new(),
            prob_dots_to_values: HashMap::new(),
            prob_dot_context: Default::default(),
            pdots_to_cdots: HashMap::new(),
        }
    }

    pub fn join(
        &mut self,
        other: &Self,
    ) -> (HashSet<u64>, HashSet<u64>, HashSet<u64>, HashSet<u64>) {
        let mut local_forgotten_dots = HashSet::new();
        let mut remote_forgotten_dots = HashSet::new();
        let mut local_fp_dots = HashSet::new();
        let mut remote_fp_dots = HashSet::new();

        // Extending the local dot mappings, so that the corresponding classical dot of every
        // probabilistic dot is known when detecting false positive.
        self.pdots_to_cdots
            .extend(other.pdots_to_cdots.iter().map(|(k, v)| (*k, v.clone())));

        // PROBABILISTIC PART

        let mut prob_dots_to_remove = HashSet::new();

        for (pdot, _) in self.prob_dots_to_values.iter() {
            // A false positive in the following APBF::query() call can happen if the remote
            // replica `other` doesn't yet know of the dot of the current iteration mapped to
            // a value at the local `self` replica.

            // In this scenario, `other` cannot have the dot mapped to a value and thus, the dot
            // entry will be later removed at `self` erroneously, provoking an exclusive element
            // in the classic `dots_to_values`.

            // In case the dot was previously known and forgotten in the meantime, `other` could
            // still have the old dot mapped to a value, which would not provoke an erroneous
            // removal. If the entry was removed meanwhile, then the dot will be correctly marked
            // for removal. Here, the query returning true helps to avoid an erroneous no-op.

            // # Whole context method
            // if !other.prob_dots_to_values.contains_key(pdot)
            //     && other.prob_dot_context.query(&pdot.to_be_bytes()) {

            // # Tombstone method
            if other.prob_dot_context.query(&pdot.to_be_bytes()) {
                prob_dots_to_remove.insert(*pdot);
                // The join of the probabilistic context may be unreliable, so inserting the dot
                // individually might help.
                self.prob_dot_context.checked_insert(&pdot.to_be_bytes());

                let cdot = self.pdots_to_cdots.get(&pdot).unwrap();

                // Possible false positive detection
                // If the dot is said to be deleted in the probabilistic structures and is not
                // deleted in the classic structures, it is either a new false positive or there
                // was an erroneous deletion exclusively from the probabilistic structures
                // somewhere along the sync chain, which can only result from a false positive.
                if !(!other.dots_to_values.contains_key(cdot) && other.dot_context.dot_in(cdot)) {
                    println!("FALSE POSITIVE AT REMOTE - PDOT: {}", pdot);
                    // Erroneous removal -> additional element in classic structure
                    remote_fp_dots.insert(*pdot);
                }
            } else {
                let cdot = self.pdots_to_cdots.get(&pdot).unwrap();
                // Check the classic context to determine if it the dot was forgotten from the remote
                // or another replica while still useful.
                if !other.dots_to_values.contains_key(cdot) && other.dot_context.dot_in(cdot) {
                    // Erroneous no-op -> additional element in probabilistic structure
                    remote_forgotten_dots.insert(*pdot);
                }
            }
        }

        for prob_dot_to_remove in prob_dots_to_remove.iter() {
            println!("REMOVING PDOT {}", prob_dot_to_remove);
            self.prob_dots_to_values.remove(prob_dot_to_remove);
        }

        for (pdot, value) in other.prob_dots_to_values.iter() {
            // A false positive in the following APBF::query() call can happen if the local replica
            // `self` doesn't yet know of the dot of the current iteration mapped to a value at the
            // remote replica `other`.

            // In this scenario, `self` will wrongly assume that it knows of the dot mapping
            // at the remote replica and will not insert it in the local probabilistic structures,
            // provoking an exclusive element in the classic `dots to values`.

            // If the local replica knew of the dot and eventually forgot it, if the dot mapping
            // of the forgotten dot still exists, it will simply be refreshed; otherwise, if it
            // has been removed in the meantime, it will erroneously be reinserted.

            // # Whole context method
            // if !self.prob_dot_context.query(&pdot.to_be_bytes()) {

            // # Tombstone method
            let in_local_prob_dot_store = self.prob_dots_to_values.contains_key(pdot);
            let in_local_prob_tombstone_set = self.prob_dot_context.query(&pdot.to_be_bytes());
            let cdot = self.pdots_to_cdots.get(&pdot).unwrap();
            let in_local_classic_dot_store = self.dots_to_values.contains_key(cdot);
            let in_local_classic_context = self.dot_context.dot_in(cdot);

            if !in_local_prob_dot_store && !in_local_prob_tombstone_set {
                self.prob_dots_to_values.insert(*pdot, value.clone());

                // If some probablistic structure along the sync chain forgot the deletion of the
                // current pdot, not propagating it properly, but it is known in the classic
                // structures
                if !in_local_classic_dot_store && in_local_classic_context {
                    // Erroneous reinsertion -> Additional element in probabilistic structure
                    local_forgotten_dots.insert(*pdot);
                }

                // # Whole context method
                // Insertion of the dot in the context, since it wasn't previously known
                // Joining the probabilistic dot contexts could be enough. However, if the
                // dot of the entry at other.prob_dots_to_values is not present in the
                // structure of `other` being joined in at `self`, some errors may occurr.
                // This way, the newly inserted element may have better chances of being
                // represented in the probabilistic dot context.
                // self.prob_dot_context.checked_insert(&pdot.to_be_bytes());

                // Possible false positive detection
                // If the dot is said to be deleted from the local probabilistic structures while
                // not deleted in the classic structures, this means that there was either a local
                // false positive or somewhere along the sync chain, one extra deletion exclusive
                // to the probabilistic structures has occurred and propagated. Since exclusive
                // deletions can only happen as a result of false positives, the root cause remains
                // a false positive.
            } else if in_local_prob_tombstone_set
                && !(!in_local_classic_dot_store && in_local_classic_context)
            {
                println!("FALSE POSITIVE AT LOCAL - PDOT: {}", pdot);
                // Erroneous no-op - additional element in classic structure
                local_fp_dots.insert(*pdot);
            }
        }

        self.prob_dot_context.union(&other.prob_dot_context);

        // CLASSIC PART

        let mut dots_to_remove = HashSet::new();

        for (cdot, _) in self.dots_to_values.iter() {
            if !other.dots_to_values.contains_key(cdot) {
                if other.dot_context.dot_in(&cdot) {
                    dots_to_remove.insert(cdot.clone());
                }
            }
            // TODO else deepjoin {join values associated with the dots - must be mergeable}
        }

        for dot_to_remove in dots_to_remove.iter() {
            self.dots_to_values.remove(dot_to_remove);
        }

        for (cdot, value) in other.dots_to_values.iter() {
            if !self.dot_context.dot_in(&cdot) {
                self.dots_to_values.insert(cdot.clone(), value.clone());
            }
        }

        self.dot_context.join(&other.dot_context);

        (
            local_forgotten_dots,
            local_fp_dots,
            remote_forgotten_dots,
            remote_fp_dots,
        )
    }

    /// Adds a value to the map, associating it with a new dot, returning the delta DotKernel
    pub fn add(&mut self, id: ID, val: T) -> Self {
        let mut delta_kernel = DotKernel {
            dots_to_values: HashMap::new(),
            dot_context: DotContext::new(),
            prob_dots_to_values: HashMap::new(),
            prob_dot_context: Default::default(),
            pdots_to_cdots: HashMap::new(),
        };
        // Creation of a dot in the underlying classic dot context
        let cdot = self.dot_context.make_dot(id);

        // Version dependent on the classic version's dot
        // A different hash function (such as BLAKE3) and a wider value (128 bits) could lessen
        // the number of collisions. However, these considerations do not seem to be important,
        // as the probabilistic dots will be mapped to the filter, which is the real cause
        // of collisions.

        // let mut hasher = DefaultHasher::new();
        // dot.hash(&mut hasher);
        // let pdot = hasher.finish();

        // Randomly generated 64 bit dot.
        let pdot: u64 = rand::thread_rng().gen();
        // # Tombstone method
        // // Creation of a dot in the underlying probabilistic dot context
        // self.prob_dot_context.checked_insert(&pdot.to_be_bytes());

        delta_kernel
            .dots_to_values
            .insert(cdot.clone(), val.clone());
        delta_kernel.dot_context.insert_dot(cdot.clone(), true);

        delta_kernel.prob_dots_to_values.insert(pdot, val.clone());
        // # Tombstone method
        // delta_kernel
        //     .prob_dot_context
        //     .checked_insert(&pdot.to_be_bytes());

        delta_kernel.pdots_to_cdots.insert(pdot, cdot.clone());

        // Registration of the mapping from the probabilistic dot to the respective classic dot of
        // associated value.
        self.pdots_to_cdots.insert(pdot, cdot.clone());
        // Insertion of the dot-to-value mapping in the classic version's map.
        self.dots_to_values.insert(cdot, val.clone());
        // Insertion of the dot-to-value mapping in the probabilistic version's map.
        self.prob_dots_to_values.insert(pdot, val);

        delta_kernel
    }

    /// Adds a value to the map, associating it with a new dot, returning the pair (dot, hash)
    /// associated with the insertion. The hash of the pair returned is a dot for the probabilistic
    /// version.
    pub fn dot_add(&mut self, id: ID, val: T) -> (Dot<ID>, u64) {
        // Create a dot in the underlying dot context
        let cdot = self.dot_context.make_dot(id);
        // Dot for the probabilistic version
        let pdot: u64 = rand::thread_rng().gen();
        // # Tombstone method
        // self.prob_dot_context.checked_insert(&pdot.to_be_bytes());

        // Insert the value, associating it with the newly created dot
        self.prob_dots_to_values.insert(pdot, val.clone());
        self.dots_to_values.insert(cdot.clone(), val);

        (cdot, pdot)
    }

    /// Removes all mappings of dots matching a value. Returns a delta dot kernel.
    pub fn remove_by_val(&mut self, val: &T) -> Self {
        let mut dots_to_remove = HashSet::new();
        let mut prob_dots_to_remove = HashSet::new();
        let mut delta_kernel = DotKernel {
            dots_to_values: HashMap::new(),
            dot_context: DotContext::new(),
            prob_dots_to_values: HashMap::new(),
            prob_dot_context: Default::default(),
            pdots_to_cdots: HashMap::new(),
        };

        // Update of the classic version's structures.
        for (cdot, value) in self.dots_to_values.iter() {
            if *val == *value {
                // The resulting delta knows of the dot
                delta_kernel.dot_context.insert_dot(cdot.clone(), false);
                // The dot is saved for later removal of its respective entry
                dots_to_remove.insert(cdot.clone());
            }
        }
        // As several dots might have been inserted, the dot context could now be compacted
        delta_kernel.dot_context.compact();

        // Update of the probabilistic version's structures.
        for (pdot, value) in self.prob_dots_to_values.iter() {
            if *val == *value {
                println!("?? PDOT: {}", pdot);
                delta_kernel
                    .prob_dot_context
                    .checked_insert(&pdot.to_be_bytes());
                prob_dots_to_remove.insert(*pdot);
                // Tombstone method
                self.prob_dot_context.checked_insert(&pdot.to_be_bytes());
            }
        }

        // Removal of the entries with the value indicated for removal
        for cdot in dots_to_remove.iter() {
            self.dots_to_values.remove(cdot);
        }
        for pdot in prob_dots_to_remove.iter() {
            self.prob_dots_to_values.remove(pdot);
        }

        delta_kernel
    }

    pub fn remove_by_dot(&mut self, dot: Dot<ID>) -> Self {
        let mut delta_kernel = DotKernel {
            dots_to_values: HashMap::new(),
            dot_context: DotContext::new(),
            prob_dots_to_values: HashMap::new(),
            prob_dot_context: Default::default(),
            pdots_to_cdots: HashMap::new(),
        };
        // If there was a value associated with the given dot...
        if self.dots_to_values.remove(&dot).is_some() {
            // The resulting delta knows of the dot
            delta_kernel.dot_context.insert_dot(dot, true);
            // delta_kernel.dot_context.insert_dot(dot, false);
            // // Compact the possible dot inserted in the dot cloud
            // delta_kernel.dot_context.compact();
        }

        delta_kernel
    }

    pub fn remove_by_prob_dot(&mut self, pdot: u64) -> Self {
        let mut delta_kernel = DotKernel {
            dots_to_values: HashMap::new(),
            dot_context: DotContext::new(),
            prob_dots_to_values: HashMap::new(),
            prob_dot_context: Default::default(),
            pdots_to_cdots: HashMap::new(),
        };

        // If there was a value associated with the given dot...
        if self.prob_dots_to_values.remove(&pdot).is_some() {
            delta_kernel
                .prob_dot_context
                .checked_insert(&pdot.to_be_bytes());
        }

        delta_kernel
    }

    pub fn remove_all(&mut self) -> Self {
        let mut delta_kernel = DotKernel {
            dots_to_values: HashMap::new(),
            dot_context: DotContext::new(),
            prob_dots_to_values: HashMap::new(),
            prob_dot_context: Default::default(),
            pdots_to_cdots: HashMap::new(),
        };

        // Iterate over the entries (specifically classic dots) and clear the map
        for (cdot, _) in self.dots_to_values.drain() {
            // Mark the dot as known in the delta kernel to be returned
            delta_kernel.dot_context.insert_dot(cdot, false);
        }
        delta_kernel.dot_context.compact();

        // Iterate over the entries (specifically probabilistic dots) and clear the map
        for (pdot, _) in self.prob_dots_to_values.drain() {
            delta_kernel
                .prob_dot_context
                .checked_insert(&pdot.to_be_bytes());
            // # Tombstone method
            self.prob_dot_context.checked_insert(&pdot.to_be_bytes());
        }

        delta_kernel
    }
}

/// Add-wins Observed-Remove Set of elements of type T and IDs of type ID
pub struct AWORSet<ID, T> {
    /// Dot kernel
    pub dot_kernel: DotKernel<ID, T>,
    /// Replica ID
    pub id: ID,
}

impl<ID, T> AWORSet<ID, T>
where
    ID: Clone + Default + Eq + Hash,
    T: Clone + Eq + Hash, // Hash is needed to collect elements into a HashSet in read queries.
{
    /// Should be used only for deltas, that should not be mutated.
    pub fn delta_new() -> Self {
        AWORSet {
            dot_kernel: DotKernel::new(),
            id: ID::default(),
        }
    }

    /// Used for mutable replicas, that need a unique ID.
    pub fn identified_new(id: ID) -> Self {
        AWORSet {
            dot_kernel: DotKernel::new(),
            id,
        }
    }

    /// Returns a reference to the underlying dot context of the data type.
    pub fn context(&self) -> &DotContext<ID> {
        &self.dot_kernel.dot_context
    }

    /// Returns a reference to the underlying probabilistic dot context of the data type.
    pub fn prob_context(&self) -> &APBF {
        &self.dot_kernel.prob_dot_context
    }

    /// Returns a HashSet of references to the values contained in the classic dots_to_values map
    /// (DotFun in the literature). They are references to the values contained in the AWORSet.
    pub fn read(&self) -> HashSet<&T> {
        self.dot_kernel.dots_to_values.values().collect()
    }

    /// Returns a HashSet of references to the values contained in the prob_dots_to_values map
    /// (DotFun in the literature). They are references to the values contained in the AWORSet.
    pub fn prob_read(&self) -> HashSet<&T> {
        self.dot_kernel.prob_dots_to_values.values().collect()
    }

    /// Returns a HashSet of clones of the values contained in the classic dots_to_values map
    /// (DotFun in the literature). They are clones of the values contained in the AWORSet.
    pub fn read_clones(&self) -> HashSet<T> {
        self.dot_kernel.dots_to_values.values().cloned().collect()
    }

    /// Returns a HashSet of clones of the values contained in the prob_dots_to_values map
    /// (DotFun in the literature). They are clones of the values contained in the AWORSet.
    pub fn prob_read_clones(&self) -> HashSet<T> {
        self.dot_kernel
            .prob_dots_to_values
            .values()
            .cloned()
            .collect()
    }

    /// Checks if the AWORSet contains the given referenced value.
    pub fn contains(&self, val: &T) -> bool {
        for value in self.dot_kernel.dots_to_values.values() {
            if *value == *val {
                return true;
            }
        }
        false
    }

    pub fn prob_contains(&self, val: &T) -> bool {
        for value in self.dot_kernel.prob_dots_to_values.values() {
            if *value == *val {
                return true;
            }
        }
        false
    }

    /// Adds a value to the AWORSet, consuming it. Returns a delta.
    pub fn add(&mut self, val: T) -> Self {
        let mut delta = AWORSet::delta_new();
        let mut delta_kernel = self.dot_kernel.remove_by_val(&val);
        delta_kernel.join(&self.dot_kernel.add(self.id.clone(), val));
        delta.dot_kernel = delta_kernel;
        delta
    }

    /// Removes a value from the AWORSet, while returning a delta.
    pub fn remove(&mut self, val: T) -> Self {
        let mut delta = AWORSet::delta_new();
        delta.dot_kernel = self.dot_kernel.remove_by_val(&val);
        delta
    }

    /// Removes all values from the AWORSet, resetting it. Returns a delta.
    pub fn reset(&mut self) -> Self {
        let mut delta = AWORSet::delta_new();
        delta.dot_kernel = self.dot_kernel.remove_all();
        delta
    }

    /// Completely clears an AWORSet - not only dot stores but also including the contexts.
    /// Not an AWORSet operation. Simply used for convenience.
    pub fn clear(&mut self) {
        self.dot_kernel.dots_to_values.clear();
        self.dot_kernel.dot_context = DotContext::new();
        self.dot_kernel.prob_dots_to_values.clear();
        self.dot_kernel.prob_dot_context = Default::default();
        self.dot_kernel.pdots_to_cdots.clear();
    }

    /// Joins two AWORSet.
    pub fn join(
        &mut self,
        other: &Self,
    ) -> (HashSet<u64>, HashSet<u64>, HashSet<u64>, HashSet<u64>) {
        self.dot_kernel.join(&other.dot_kernel)
    }

    /// Checks if the data for the probabilistic structure matches the data of the classical version.
    pub fn check_eq(&self) -> bool {
        self.read() == self.prob_read()
    }

    /// Returns the pair of dots for a given value.
    pub fn pdot_of_value(&self, value: &T) -> Option<u64> {
        let mut res = None;
        for (pdot, v) in self.dot_kernel.prob_dots_to_values.iter() {
            if v == value {
                res = Some(*pdot);
                break;
            }
        }
        if res == None {
            for (active_cdot, v) in self.dot_kernel.dots_to_values.iter() {
                if v == value {
                    for (pdot, cdot) in self.dot_kernel.pdots_to_cdots.iter() {
                        if active_cdot == cdot {
                            res = Some(*pdot);
                            break;
                        }
                    }
                    break;
                }
            }
        }
        res
    }
}

/// Multi-value register (apparently optimized)
pub struct MVReg<ID, T> {
    /// Dot kernel
    pub dot_kernel: DotKernel<ID, T>,
    /// Replica ID
    pub id: ID,
}

impl<ID, T> MVReg<ID, T>
where
    ID: Clone + Default + Eq + Hash,
    // Ord aids MVReg::resolve(), as it enables a simple join to be defined.
    T: Clone + Default + Eq + Hash + Ord,
{
    /// Should be used only for deltas, that should not be mutated.
    pub fn delta_new() -> Self {
        MVReg {
            dot_kernel: DotKernel::new(),
            id: ID::default(),
        }
    }

    /// Used for mutable replicas, that need a unique ID.
    pub fn identified_new(id: ID) -> Self {
        MVReg {
            dot_kernel: DotKernel::new(),
            id,
        }
    }

    /// Returns a reference to the underlying dot context of the data type.
    pub fn context(&self) -> &DotContext<ID> {
        &self.dot_kernel.dot_context
    }

    /// Writes a value to the register. Returns the delta.
    pub fn write(&mut self, val: T) -> Self {
        let mut delta = Self::delta_new();
        // Remove all previous dots and associated values
        let mut delta_kernel = self.dot_kernel.remove_all();
        // Add the value to the dot kernel of the MVReg and join the delta with the one that
        // resulted from the clearance of the register.
        delta_kernel.join(&self.dot_kernel.add(self.id.clone(), val));
        delta.dot_kernel = delta_kernel;
        delta
    }

    /// Returns a HashSet of references to the values contained in the classic dots_to_values map
    /// (DotFun in the literature). They are references to the value(s) of the MVReg.
    pub fn read(&self) -> HashSet<&T> {
        self.dot_kernel.dots_to_values.values().collect()
    }

    /// Returns a HashSet of references to the values contained in the prob_dots_to_values map
    /// (DotFun in the literature). They are references to the value(s) of the MVReg.
    pub fn prob_read(&self) -> HashSet<&T> {
        self.dot_kernel.prob_dots_to_values.values().collect()
    }

    /// Checks if the data for the probabilistic structure matches the data of the classical version.
    pub fn check_eq(&self) -> bool {
        self.read() == self.prob_read()
    }

    /// Returns a HashSet of clones of the values contained in the classic dots_to_values map
    /// (DotFun in the literature). They are clones of the value(s) of the MVReg.
    pub fn read_clones(&self) -> HashSet<T> {
        self.dot_kernel.dots_to_values.values().cloned().collect()
    }

    /// Returns a HashSet of clones of the values contained in the prob_dots_to_values map
    /// (DotFun in the literature). They are clones of the value(s) of the MVReg.
    pub fn prob_read_clones(&self) -> HashSet<T> {
        self.dot_kernel
            .prob_dots_to_values
            .values()
            .cloned()
            .collect()
    }

    /// Resets the register by clearing its values. Returns the delta.
    pub fn reset(&mut self) -> Self {
        let mut delta = MVReg::delta_new();
        delta.dot_kernel = self.dot_kernel.remove_all();
        delta
    }

    /// Removes every value in the set that is not a maximal.
    pub fn resolve(&mut self) -> Self {
        let mut delta = MVReg::delta_new();
        let mut dots_to_remove = HashSet::new();
        let mut prob_dots_to_remove = HashSet::new();

        // Classic version

        // Finding the maximal value in the register.
        let maximal = self
            .dot_kernel
            .dots_to_values
            .values()
            .max()
            .unwrap_or(&T::default())
            .clone();
        // An iterator that will return only the (dot, value) pairs of values smaller than the maximal.
        let filtered = self
            .dot_kernel
            .dots_to_values
            .iter()
            .filter(|(_, v)| **v < maximal);
        // Mark the dot for removal.
        for (d, _) in filtered {
            dots_to_remove.insert(d.clone());
        }
        // Remove the dot and join the resulting smaller delta into the delta to be returned.
        for d in dots_to_remove.drain() {
            delta.dot_kernel.join(&self.dot_kernel.remove_by_dot(d));
        }

        // Probabilistic version

        // Finding the maximal value in the register.
        let maximal = self
            .dot_kernel
            .prob_dots_to_values
            .values()
            .max()
            .unwrap_or(&T::default())
            .clone();
        // An iterator that will return only the (dot, value) pairs of values smaller than the maximal.
        let filtered = self
            .dot_kernel
            .prob_dots_to_values
            .iter()
            .filter(|(_, v)| **v < maximal);
        // Mark the dot for removal.
        for (d, _) in filtered {
            prob_dots_to_remove.insert(*d);
        }
        // Remove the dot and join the resulting smaller delta into the delta to be returned.
        for pdot in prob_dots_to_remove.drain() {
            delta
                .dot_kernel
                .join(&self.dot_kernel.remove_by_prob_dot(pdot));
        }

        delta
    }

    /// Joins the dot kernels of a `self` and an `other` MVReg.
    pub fn join(&mut self, other: &Self) {
        self.dot_kernel.join(&other.dot_kernel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_testing() {
        let mut dc = DotContext::new();
        dc.causal_context.insert(0, 0);
        dc.causal_context.insert(1, 0);
        assert_eq!(dc.causal_context.insert(0, 1), Some(0));
        dc.dot_cloud.insert((0, 0));
        dc.dot_cloud.insert((1, 0));
        dc.dot_cloud.insert((0, 1));
        dc.dot_cloud.insert((2, 0));
        assert_eq!(dc.dot_in(&(0, 0)), true);
        assert_eq!(dc.dot_in(&(1, 0)), true);
        assert_eq!(dc.causal_context.get(&2), None);
        assert_eq!(dc.dot_in(&(2, 0)), true);
        assert_ne!(dc.dot_in(&(2, 1)), true);
        assert_eq!(dc.dot_in(&(0, 1)), true);
    }

    #[test]
    fn compaction() {
        let mut dc = DotContext::new();
        dc.dot_cloud.insert((0, 1));
        dc.dot_cloud.insert((0, 3));
        assert_eq!(dc.causal_context.len(), 0);
        dc.compact();
        assert_ne!(dc.causal_context.len(), 0, "{:?}", dc.causal_context);
        assert_ne!(dc.dot_cloud.len(), 0, "{:?}", dc.dot_cloud);
    }

    #[test]
    fn dot_making() {
        let mut dc = DotContext::new();
        dc.make_dot(1);
        dc.make_dot(1);
        dc.make_dot(2);
        assert_eq!(dc.causal_context.get(&1), Some(&2));
        assert_eq!(dc.causal_context.get(&2), Some(&1));
        assert_eq!(dc.causal_context.get(&3), None);
    }

    #[test]
    fn context_joining() {
        let mut local = DotContext::new();
        let mut remote = DotContext::new();
        local.make_dot(0);
        local.make_dot(0);
        remote.make_dot(0);
        remote.make_dot(1);
        remote.make_dot(1);
        remote.make_dot(1);
        local.make_dot(2);
        local.join(&remote);
        assert_eq!(local.causal_context.get(&0), Some(&2));
        assert_eq!(local.causal_context.get(&1), Some(&3));
        assert_eq!(local.causal_context.get(&2), Some(&1));
    }

    #[test]
    fn aworset_testing() {
        let mut a_set = AWORSet::identified_new(1);
        let mut b_set = AWORSet::identified_new(2);
        assert_eq!(a_set.contains(&4), false);
        assert_eq!(b_set.contains(&2), false);
        a_set.add(4);
        b_set.add(2);
        assert_eq!(a_set.contains(&2), false);
        assert_eq!(a_set.contains(&4), true);
        assert_eq!(b_set.contains(&2), true);
        assert_eq!(b_set.contains(&4), false);
        a_set.join(&b_set);
        b_set.join(&a_set);
        assert_eq!(a_set.contains(&2), true);
        assert_eq!(a_set.contains(&4), true);
        assert_eq!(b_set.contains(&2), true);
        assert_eq!(b_set.contains(&4), true);
        // Extra value
        assert_eq!(b_set.contains(&3), false);
        b_set.remove(2);
        assert_eq!(b_set.contains(&2), false);
        assert_eq!(b_set.contains(&4), true);
        a_set.join(&b_set);
        assert_eq!(a_set.contains(&2), false);
        assert_eq!(a_set.contains(&4), true);
        a_set.reset();
        assert_eq!(a_set.contains(&2), false);
        assert_eq!(a_set.contains(&4), false);
    }

    #[test]
    fn add_wins() {
        let mut a_set = AWORSet::identified_new(1);
        let mut b_set = AWORSet::identified_new(2);
        a_set.add(1);
        b_set.add(1);
        assert_eq!(a_set.contains(&1), true);
        assert_eq!(b_set.contains(&1), true);
        b_set.remove(1);
        assert_eq!(b_set.contains(&1), false);
        a_set.join(&b_set);
        assert_eq!(a_set.contains(&1), true);
        assert_eq!(b_set.contains(&1), false);
        b_set.join(&a_set);
        assert_eq!(a_set.contains(&1), true);
        assert_eq!(b_set.contains(&1), true);
    }

    #[test]
    fn mvreg_test() {
        let mut a_reg = MVReg::identified_new(1);
        let mut b_reg = MVReg::identified_new(2);

        let mut delta1 = a_reg.write("hello");
        delta1.join(&a_reg.write("world"));

        let mut delta2 = b_reg.write("world");
        delta2.join(&b_reg.write("hello"));

        // The set of expected values in the register
        let mut cmp_set = HashSet::with_capacity(2);
        cmp_set.insert("world");
        cmp_set.insert("hello");

        // Testing joining the deltas
        delta1.join(&delta2);
        // Testing joining full states of replicas
        a_reg.join(&b_reg);

        assert_eq!(delta1.read_clones(), cmp_set);
        assert_eq!(a_reg.read_clones(), cmp_set);
    }
}
