use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::sync::{Semaphore, SemaphorePermit};

pub struct AdmissionQueue {
    semaphore: Semaphore,
    waiting: AtomicUsize,
}

pub struct AdmissionTicket<'a> {
    _permit: SemaphorePermit<'a>,
}

impl AdmissionQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            semaphore: Semaphore::new(capacity),
            waiting: AtomicUsize::new(0),
        }
    }

    pub fn depth(&self) -> usize {
        self.waiting.load(Ordering::SeqCst)
    }

    pub async fn admit(&self) -> AdmissionTicket<'_> {
        self.waiting.fetch_add(1, Ordering::SeqCst);
        let permit = self
            .semaphore
            .acquire()
            .await
            .expect("semaphore is never closed");
        self.waiting.fetch_sub(1, Ordering::SeqCst);
        AdmissionTicket { _permit: permit }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize as StdAtomicUsize, Ordering as StdOrdering};
    use std::sync::Arc;

    use super::*;

    #[tokio::test]
    async fn admit_serializes_single_capacity_access() {
        let queue = Arc::new(AdmissionQueue::new(1));
        let concurrent = Arc::new(StdAtomicUsize::new(0));
        let max_concurrent = Arc::new(StdAtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..5 {
            let queue = queue.clone();
            let concurrent = concurrent.clone();
            let max_concurrent = max_concurrent.clone();
            handles.push(tokio::spawn(async move {
                let _ticket = queue.admit().await;
                let now = concurrent.fetch_add(1, StdOrdering::SeqCst) + 1;
                max_concurrent.fetch_max(now, StdOrdering::SeqCst);
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                concurrent.fetch_sub(1, StdOrdering::SeqCst);
            }));
        }
        for handle in handles {
            handle.await.expect("task did not panic");
        }

        assert_eq!(max_concurrent.load(StdOrdering::SeqCst), 1);
    }

    #[tokio::test]
    async fn depth_reflects_waiting_tasks() {
        let queue = Arc::new(AdmissionQueue::new(1));
        let held = queue.admit().await;

        let queue2 = queue.clone();
        let waiter = tokio::spawn(async move {
            let _ticket = queue2.admit().await;
        });

        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert_eq!(queue.depth(), 1);

        drop(held);
        waiter.await.expect("task did not panic");
    }
}
