import XCTest
@testable import <#AppModule#>

@MainActor
final class <#FeatureName#>ViewModelTests: XCTestCase {
    func test_whenLoadSuccess_thenStateBecomesSuccess() async throws {
        let service = MockService(result: .success(<#SuccessPayload#>))
        let sut = <#FeatureViewModel#>(service: service)

        await sut.load()

        XCTAssertEqual(sut.state, .success(<#ExpectedValue#>))
    }

    func test_whenLoadFails_thenStateBecomesError() async {
        let service = MockService(result: .failure(MockError.expected))
        let sut = <#FeatureViewModel#>(service: service)

        await sut.load()

        XCTAssertEqual(sut.state, .error(<#ExpectedErrorValue#>))
    }

    func test_whenTaskIsCancelled_thenLoadingStopsSafely() async {
        let service = MockService(result: .pending)
        let sut = <#FeatureViewModel#>(service: service)

        let task = Task { await sut.load() }
        task.cancel()
        _ = await task.result

        XCTAssertFalse(sut.isLoading)
    }

    func test_whenRequiredFieldsAreMissing_thenSaveActionStaysDisabled() {
        let sut = <#FeatureViewModel#>(service: MockService(result: .success(<#SuccessPayload#>)))

        sut.<#requiredField#> = <#invalidValue#>

        XCTAssertFalse(sut.<#isSaveEnabled#>)
    }

    func test_whenNothingIsSelected_thenBatchActionStaysDisabled() {
        let sut = <#FeatureViewModel#>(service: MockService(result: .success(<#SuccessPayload#>)))

        sut.<#selectedItems#> = []

        XCTAssertFalse(sut.<#isBatchActionEnabled#>)
    }
}

private struct MockService: <#ServiceProtocol#> {
    let result: MockResult

    enum MockResult {
        case success(<#PayloadType#>)
        case failure(Error)
        case pending
    }

    func <#loadMethod#>() async throws -> <#PayloadType#> {
        switch result {
        case let .success(payload):
            return payload
        case let .failure(error):
            throw error
        case .pending:
            try await Task.sleep(nanoseconds: 50_000_000)
            try Task.checkCancellation()
            throw CancellationError()
        }
    }
}

private enum MockError: Error {
    case expected
}
