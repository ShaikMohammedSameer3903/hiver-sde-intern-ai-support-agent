import json

def main():
    with open('artifacts/data_profile.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    meta = data['dataset_metadata']
    print("=" * 95)
    print("CUSTOMER SUPPORT ON TWITTER — DATASET PROFILE SUMMARY")
    print("=" * 95)
    print(f"Total Tweets Processed : {meta['total_rows']:,}")
    print(f"Inbound (Customer)     : {meta['inbound_count']:,} ({meta['inbound_ratio']*100:.2f}%)")
    print(f"Outbound (Brand)       : {meta['outbound_count']:,} ({(1-meta['inbound_ratio'])*100:.2f}%)")
    print(f"Unique Brands          : {meta['unique_brands_count']}")
    print(f"Date Range             : {meta['date_range']['start']} to {meta['date_range']['end']}")
    print(f"Average Text Length    : {meta['text_statistics']['char_length_mean']} chars (~{meta['text_statistics']['word_count_mean']} words)")
    print(f"Median Text Length     : {meta['text_statistics']['char_length_median']} chars (~{meta['text_statistics']['word_count_median']} words)")
    print()
    print("=" * 95)
    print("TOP BRAND CANDIDATES RANKING TABLE")
    print("=" * 95)
    header = f"{'Brand':<16} | {'Total Msgs':<11} | {'Brand Replies':<13} | {'Cust Msgs':<10} | {'Conversations':<13} | {'Res-Rich':<10} | {'Suitability':<11}"
    print(header)
    print("-" * len(header))
    for b in data['top_brand_rankings'][:15]:
        row = f"{b['brand']:<16} | {b['total_messages']:<11,d} | {b['outbound_brand_replies']:<13,d} | {b['inbound_customer_msgs']:<10,d} | {b['estimated_conversations']:<13,d} | {b['resolution_rich_replies']:<10,d} | {b['suitability_score']:<11.2f}"
        print(row)
    print("=" * 95)

if __name__ == '__main__':
    main()
