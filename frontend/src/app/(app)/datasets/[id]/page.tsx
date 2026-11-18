import { Suspense } from "react";

import { DatasetDetail } from "@/components/DatasetDetail";

async function Detail({ params }: { params: PageProps<"/datasets/[id]">["params"] }) {
  const { id } = await params;
  return <DatasetDetail id={id} />;
}

export default function DatasetPage({ params }: PageProps<"/datasets/[id]">) {
  // With Cache Components, request-time data (route params) must be read under Suspense.
  return (
    <Suspense>
      <Detail params={params} />
    </Suspense>
  );
}
